#include "actor/mcts/runtime/inference_runtime.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lc0/move_index.h"

namespace engine {
namespace {

struct InflightEval {
  std::unique_ptr<GameTask> task;
  EvalRequest request;
  EvalResponse response;
  std::vector<bool> completed_items;
  std::size_t remaining_items = 0;
};

struct RouteEntry {
  InflightEval* inflight = nullptr;
  std::size_t request_item_idx = 0;
};

InflightEval MakeInflight(PendingEval pending) {
  assert(pending.task != nullptr);
  assert(pending.task->state == TaskState::kWaitingEval);
  assert(!pending.task->response.has_value());
  assert(!pending.request.items.empty());

  InflightEval inflight{
      .task = std::move(pending.task),
      .request = std::move(pending.request),
  };
  inflight.response.items.resize(inflight.request.items.size());
  inflight.completed_items.assign(inflight.request.items.size(), false);
  inflight.remaining_items = inflight.request.items.size();
  return inflight;
}

template <class Clock, class Duration>
void GatherUntil(ThreadSafeQueue<PendingEval>& request_queue,
                 std::chrono::time_point<Clock, Duration> deadline,
                 std::deque<InflightEval>* inflight) {
  while (auto next = request_queue.pop_until(deadline)) {
    inflight->push_back(MakeInflight(std::move(*next)));
  }
}

int BuildBatchForSlot(
    std::deque<InflightEval>* inflight, int slot, int slot_count,
    int max_batch_size, std::vector<RouteEntry>* route,
    std::vector<std::span<const lczero::Position>>* batch_positions) {
  assert(slot >= 0);
  assert(slot < slot_count);
  assert(slot_count > 0);
  assert(max_batch_size > 0);

  route->clear();
  route->reserve(static_cast<std::size_t>(max_batch_size));
  batch_positions->clear();
  batch_positions->reserve(static_cast<std::size_t>(max_batch_size));

  int batch_size = 0;
  for (InflightEval& eval : *inflight) {
    for (std::size_t item_idx = 0; item_idx < eval.request.items.size();
         ++item_idx) {
      if (batch_size == max_batch_size) break;
      if (eval.completed_items[item_idx]) continue;

      const EvalRequestItem& item = eval.request.items[item_idx];
      assert(item.len > 0);
      assert(item.len <= item.positions.size());
      const int requested_slot = eval.task->BackendSlotForRequestItem(item);
      if (requested_slot < 0 || requested_slot >= slot_count) continue;
      if (requested_slot != slot) continue;

      route->push_back(RouteEntry{
          .inflight = &eval,
          .request_item_idx = item_idx,
      });
      batch_positions->push_back(
          std::span<const lczero::Position>(item.positions.data(), item.len));
      ++batch_size;
    }
    if (batch_size == max_batch_size) break;
  }

  return batch_size;
}

void ScatterBatch(const std::vector<RouteEntry>& route,
                  const InferenceOutputs& outputs, int policy_size) {
  assert(policy_size > 0);
  assert(outputs.policy_logits.size() ==
         route.size() * static_cast<std::size_t>(policy_size));
  assert(outputs.wdl_probs.size() == route.size() * 3);

  const std::size_t policy_stride = static_cast<std::size_t>(policy_size);
  for (std::size_t batch_idx = 0; batch_idx < route.size(); ++batch_idx) {
    const RouteEntry& entry = route[batch_idx];
    assert(entry.inflight != nullptr);
    assert(entry.request_item_idx < entry.inflight->response.items.size());

    EvalResponseItem& item =
        entry.inflight->response.items[entry.request_item_idx];
    const std::size_t policy_begin = batch_idx * policy_stride;
    item.policy_logits.assign(
        outputs.policy_logits.begin() + policy_begin,
        outputs.policy_logits.begin() + policy_begin + policy_stride);

    const std::size_t wdl_begin = batch_idx * 3;
    item.wdl_probs.assign(outputs.wdl_probs.begin() + wdl_begin,
                          outputs.wdl_probs.begin() + wdl_begin + 3);

    entry.inflight->completed_items[entry.request_item_idx] = true;
    --entry.inflight->remaining_items;
  }
}

}  // namespace

InferenceRuntime::InferenceRuntime(InferenceChannels channels,
                                   std::shared_ptr<InferenceBackend> backend,
                                   const FeatureEncoder* encoder,
                                   InferenceRuntimeOptions options)
    : InferenceRuntime(channels,
                       std::vector<std::shared_ptr<InferenceBackend>>{
                           std::move(backend),
                       },
                       encoder, options) {}

InferenceRuntime::InferenceRuntime(
    InferenceChannels channels,
    std::vector<std::shared_ptr<InferenceBackend>> backends,
    const FeatureEncoder* encoder, InferenceRuntimeOptions options)
    : channels_(channels),
      backends_(std::move(backends)),
      encoder_(encoder),
      options_(options) {
  assert(channels_.request_queue != nullptr);
  assert(channels_.ready_queue != nullptr);
  assert(!backends_.empty());
  for (const auto& backend : backends_) {
    assert(backend != nullptr);
  }
  assert(encoder_ != nullptr);
  assert(options_.max_batch_size > 0);
  assert(options_.flush_timeout.count() >= 0);
}

InferenceRuntime::~InferenceRuntime() { Stop(); }

void InferenceRuntime::Start() {
  std::lock_guard<std::mutex> lock(state_mu_);
  if (started_) return;
  if (stopped_) return;

  for (const auto& backend : backends_) {
    const Status status = backend->Load();
    assert(status.ok());
  }

  started_ = true;
  inference_thread_ = std::thread(&InferenceRuntime::RunLoop, this);
}

void InferenceRuntime::Stop() {
  bool should_stop = false;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (!started_ || stopped_) return;
    stopped_ = true;
    should_stop = true;
  }
  if (!should_stop) return;

  // Stop closes the inbound request queue to wake the blocked inference thread.
  // The coordinator must stop workers first so no worker can still push new
  // PendingEval items after inference shutdown begins.
  channels_.request_queue->close();
  if (inference_thread_.joinable()) inference_thread_.join();
}

void InferenceRuntime::RunLoop() {
  // This loop treats request_queue as the single inbound wakeup source.
  // Because Stop() closes that queue, worker shutdown must happen before
  // inference shutdown to preserve the request handoff invariant.
  std::deque<InflightEval> inflight;
  std::vector<RouteEntry> route;
  std::vector<std::span<const lczero::Position>> batch_positions;
  std::vector<uint8_t> planes_buffer(
      static_cast<std::size_t>(options_.max_batch_size) * kInputElements);
  InferenceOutputs outputs;

  while (true) {
    if (inflight.empty()) {
      auto first = channels_.request_queue->pop();
      if (!first) return;

      inflight.push_back(MakeInflight(std::move(*first)));
      GatherUntil(*channels_.request_queue,
                  std::chrono::steady_clock::now() + options_.flush_timeout,
                  &inflight);
    } else {
      // A partially processed request is already waiting, so do not add extra
      // latency. Just drain any currently available requests into local state.
      GatherUntil(*channels_.request_queue, std::chrono::steady_clock::now(),
                  &inflight);
    }

    bool ran_batch = false;
    const int slot_count = static_cast<int>(backends_.size());
    for (int slot = 0; slot < slot_count; ++slot) {
      const int batch_size =
          BuildBatchForSlot(&inflight, slot, slot_count,
                            options_.max_batch_size, &route, &batch_positions);
      if (batch_size == 0) continue;

      ran_batch = true;
      encoder_->EncodeBatch(
          batch_positions,
          std::span<uint8_t>(
              planes_buffer.data(),
              static_cast<std::size_t>(batch_size) * kInputElements));

      const Status status = backends_[static_cast<std::size_t>(slot)]->Run(
          InferenceBatch{
              .planes = planes_buffer.data(),
              .batch_size = batch_size,
          },
          &outputs);
      assert(status.ok());

      ScatterBatch(route, outputs, lczero::kPolicySize);
    }

    if (!ran_batch) {
      throw std::runtime_error(
          "inference runtime found unroutable eval request");
    }

    while (!inflight.empty() && inflight.front().remaining_items == 0) {
      InflightEval completed = std::move(inflight.front());
      inflight.pop_front();
      completed.task->response.emplace(std::move(completed.response));
      completed.task->state = TaskState::kReady;
      const bool pushed =
          channels_.ready_queue->push(std::move(completed.task));
      assert(pushed);
    }
  }
}

}  // namespace engine
