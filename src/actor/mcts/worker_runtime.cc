#include "worker_runtime.h"

#include <cassert>
#include <utility>

namespace engine {

WorkerRuntime::WorkerRuntime(int num_workers, WorkerChannels channels)
    : num_workers_(num_workers), channels_(channels) {
  assert(num_workers_ > 0);
  assert(channels_.ready_queue != nullptr);
  assert(channels_.request_queue != nullptr);
  assert(channels_.completion_queue != nullptr);
}

WorkerRuntime::~WorkerRuntime() { Stop(); }

void WorkerRuntime::Start() {
  std::lock_guard<std::mutex> lock(state_mu_);
  if (started_) return;
  if (stopped_) return;

  started_ = true;
  workers_.reserve(static_cast<size_t>(num_workers_));
  for (int i = 0; i < num_workers_; ++i) {
    workers_.emplace_back(&WorkerRuntime::WorkerLoop, this);
  }
}

void WorkerRuntime::Stop() {
  bool should_stop = false;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (!started_ || stopped_) return;
    stopped_ = true;
    should_stop = true;
  }
  if (!should_stop) return;

  channels_.ready_queue->close();
  for (std::thread& worker : workers_) {
    if (worker.joinable()) worker.join();
  }
  workers_.clear();
}

void WorkerRuntime::WorkerLoop() {
  while (auto task = channels_.ready_queue->pop()) {
    assert(*task != nullptr);
    assert((*task)->searcher != nullptr);

    std::optional<EvalRequest> request;
    switch ((*task)->state) {
      case TaskState::kNew:
        assert(!(*task)->coroutine.has_value());
        (*task)->coroutine.emplace((*task)->searcher->Run());
        request = (*task)->coroutine->next();
        break;

      case TaskState::kReady:
        assert((*task)->coroutine.has_value());
        assert((*task)->response.has_value());
        request = (*task)->coroutine->send(std::move(*(*task)->response));
        (*task)->response.reset();
        break;

      case TaskState::kWaitingEval:
      default:
        assert(false);
        continue;
    }

    if (request.has_value()) {
      (*task)->state = TaskState::kWaitingEval;
      const bool pushed = channels_.request_queue->push(
          PendingEval{std::move(*task), std::move(*request)});
      assert(pushed);
      continue;
    }

    assert((*task)->coroutine->done());
    SearchResult result = (*task)->coroutine->take_result();
    const bool pushed = channels_.completion_queue->push(
        CompletedSearch{std::move(*task), std::move(result)});
    assert(pushed);
  }
}

}  // namespace engine
