#include "actor/mcts/runtime/inference_runtime.h"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "actor/mcts/runtime/match_types.h"
#include "lc0/move_index.h"

namespace engine {
namespace {

using namespace std::chrono_literals;

class TaggedGameTask final : public GameTask {
 public:
  explicit TaggedGameTask(int task_id) : task_id(task_id) {}
  int task_id = 0;
};

class RecordingBackend final : public InferenceBackend {
 public:
  Status Load() override {
    ++load_calls_;
    return Status::Ok();
  }

  Status Run(const InferenceBatch& batch, InferenceOutputs* out) override {
    if (out == nullptr) return Status::Error("output buffer is null");
    if (load_calls_ <= 0) return Status::Error("backend not loaded");

    batch_sizes_.push_back(batch.batch_size);
    out->policy_logits.clear();
    out->wdl_probs.clear();

    for (int batch_idx = 0; batch_idx < batch.batch_size; ++batch_idx) {
      const int item_id = next_item_id_++;
      for (int policy_idx = 0; policy_idx < lczero::kPolicySize; ++policy_idx) {
        out->policy_logits.push_back(
            static_cast<float>(item_id * 10 + policy_idx));
      }
      out->wdl_probs.push_back(static_cast<float>(item_id));
      out->wdl_probs.push_back(static_cast<float>(item_id + 100));
      out->wdl_probs.push_back(static_cast<float>(item_id + 200));
    }

    return Status::Ok();
  }

  std::string Name() const override { return "recording"; }

  int load_calls() const { return load_calls_; }
  const std::vector<int>& batch_sizes() const { return batch_sizes_; }

 private:
  int load_calls_ = 0;
  int next_item_id_ = 0;
  std::vector<int> batch_sizes_;
};

class MarkerBackend final : public InferenceBackend {
 public:
  explicit MarkerBackend(int marker) : marker_(marker) {}

  Status Load() override {
    ++load_calls_;
    return Status::Ok();
  }

  Status Run(const InferenceBatch& batch, InferenceOutputs* out) override {
    if (out == nullptr) return Status::Error("output buffer is null");

    batch_sizes_.push_back(batch.batch_size);
    out->policy_logits.clear();
    out->wdl_probs.clear();
    for (int batch_idx = 0; batch_idx < batch.batch_size; ++batch_idx) {
      const int item_value = marker_ + next_item_id_++;
      for (int policy_idx = 0; policy_idx < lczero::kPolicySize; ++policy_idx) {
        out->policy_logits.push_back(static_cast<float>(item_value));
      }
      out->wdl_probs.push_back(static_cast<float>(item_value));
      out->wdl_probs.push_back(static_cast<float>(item_value + 100));
      out->wdl_probs.push_back(static_cast<float>(item_value + 200));
    }
    return Status::Ok();
  }

  std::string Name() const override { return "marker"; }

  int load_calls() const { return load_calls_; }
  const std::vector<int>& batch_sizes() const { return batch_sizes_; }

 private:
  int marker_ = 0;
  int load_calls_ = 0;
  int next_item_id_ = 0;
  std::vector<int> batch_sizes_;
};

PendingEval MakePendingEval(int task_id, std::size_t num_items) {
  auto task = std::make_unique<TaggedGameTask>(task_id);
  task->state = TaskState::kWaitingEval;

  EvalRequest request;
  request.items.resize(num_items);
  for (EvalRequestItem& item : request.items) {
    item.len = 1;
    item.positions[0] = lczero::Position();
  }

  return PendingEval{
      .task = std::unique_ptr<GameTask>(std::move(task)),
      .request = std::move(request),
  };
}

lczero::Position WhiteToMovePosition() {
  return lczero::Position::FromFen(lczero::ChessBoard::kStartposFen);
}

lczero::Position BlackToMovePosition() {
  return lczero::Position::FromFen(
      "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
}

PendingEval MakeRoutedPendingEval(int white_backend_slot,
                                  int black_backend_slot,
                                  bool active_player_is_white,
                                  std::vector<bool> black_to_move_items) {
  auto task = std::make_unique<MatchTask>();
  task->state = TaskState::kWaitingEval;
  task->white_backend_slot = white_backend_slot;
  task->black_backend_slot = black_backend_slot;
  task->active_player_is_white = active_player_is_white;

  EvalRequest request;
  request.items.resize(black_to_move_items.size());
  for (std::size_t i = 0; i < black_to_move_items.size(); ++i) {
    EvalRequestItem& item = request.items[i];
    item.len = 1;
    item.positions[0] =
        black_to_move_items[i] ? BlackToMovePosition() : WhiteToMovePosition();
  }

  return PendingEval{
      .task = std::unique_ptr<GameTask>(std::move(task)),
      .request = std::move(request),
  };
}

std::unique_ptr<TaggedGameTask> PopReadyTask(
    ThreadSafeQueue<std::unique_ptr<GameTask>>& ready_queue) {
  std::optional<std::unique_ptr<GameTask>> task = ready_queue.pop();
  EXPECT_TRUE(task.has_value());
  EXPECT_TRUE(*task != nullptr);
  auto* tagged = dynamic_cast<TaggedGameTask*>(task->get());
  EXPECT_NE(tagged, nullptr);
  return std::unique_ptr<TaggedGameTask>(
      static_cast<TaggedGameTask*>(task->release()));
}

void ExpectItemPayload(const EvalResponseItem& item, int expected_item_id) {
  ASSERT_EQ(item.policy_logits.size(),
            static_cast<std::size_t>(lczero::kPolicySize));
  ASSERT_EQ(item.wdl_probs.size(), 3u);
  EXPECT_FLOAT_EQ(item.policy_logits[0],
                  static_cast<float>(expected_item_id * 10));
  EXPECT_FLOAT_EQ(item.policy_logits[1],
                  static_cast<float>(expected_item_id * 10 + 1));
  EXPECT_FLOAT_EQ(item.policy_logits[2],
                  static_cast<float>(expected_item_id * 10 + 2));
  EXPECT_FLOAT_EQ(
      item.policy_logits.back(),
      static_cast<float>(expected_item_id * 10 + (lczero::kPolicySize - 1)));
  EXPECT_FLOAT_EQ(item.wdl_probs[0], static_cast<float>(expected_item_id));
  EXPECT_FLOAT_EQ(item.wdl_probs[1],
                  static_cast<float>(expected_item_id + 100));
  EXPECT_FLOAT_EQ(item.wdl_probs[2],
                  static_cast<float>(expected_item_id + 200));
}

std::unique_ptr<MatchTask> PopReadyMatchTask(
    ThreadSafeQueue<std::unique_ptr<GameTask>>& ready_queue) {
  std::optional<std::unique_ptr<GameTask>> task = ready_queue.pop();
  EXPECT_TRUE(task.has_value());
  EXPECT_TRUE(*task != nullptr);
  auto* match_task = dynamic_cast<MatchTask*>(task->get());
  EXPECT_NE(match_task, nullptr);
  return std::unique_ptr<MatchTask>(static_cast<MatchTask*>(task->release()));
}

void ExpectMarker(const EvalResponseItem& item, int expected_value) {
  ASSERT_EQ(item.policy_logits.size(),
            static_cast<std::size_t>(lczero::kPolicySize));
  ASSERT_EQ(item.wdl_probs.size(), 3u);
  EXPECT_FLOAT_EQ(item.policy_logits[0], static_cast<float>(expected_value));
  EXPECT_FLOAT_EQ(item.policy_logits.back(),
                  static_cast<float>(expected_value));
  EXPECT_FLOAT_EQ(item.wdl_probs[0], static_cast<float>(expected_value));
  EXPECT_FLOAT_EQ(item.wdl_probs[1], static_cast<float>(expected_value + 100));
  EXPECT_FLOAT_EQ(item.wdl_probs[2], static_cast<float>(expected_value + 200));
}

TEST(InferenceRuntime, BatchesAcrossMultipleRequestsAndRequeuesReadyTasks) {
  ThreadSafeQueue<PendingEval> request_queue;
  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue;
  auto backend = std::make_shared<RecordingBackend>();
  const FeatureEncoder encoder;

  InferenceRuntime runtime(
      InferenceChannels{
          .request_queue = &request_queue,
          .ready_queue = &ready_queue,
      },
      backend, &encoder,
      InferenceRuntimeOptions{
          .max_batch_size = 8,
          .flush_timeout = 5ms,
      });
  runtime.Start();

  ASSERT_TRUE(
      request_queue.push(MakePendingEval(/*task_id=*/1, /*num_items=*/1)));
  ASSERT_TRUE(
      request_queue.push(MakePendingEval(/*task_id=*/2, /*num_items=*/2)));

  std::unique_ptr<TaggedGameTask> first = PopReadyTask(ready_queue);
  std::unique_ptr<TaggedGameTask> second = PopReadyTask(ready_queue);

  runtime.Stop();

  ASSERT_TRUE(first->response.has_value());
  ASSERT_TRUE(second->response.has_value());
  EXPECT_EQ(first->state, TaskState::kReady);
  EXPECT_EQ(second->state, TaskState::kReady);
  EXPECT_EQ(first->task_id, 1);
  EXPECT_EQ(second->task_id, 2);
  ASSERT_EQ(first->response->items.size(), 1u);
  ASSERT_EQ(second->response->items.size(), 2u);
  ExpectItemPayload(first->response->items[0], /*expected_item_id=*/0);
  ExpectItemPayload(second->response->items[0], /*expected_item_id=*/1);
  ExpectItemPayload(second->response->items[1], /*expected_item_id=*/2);
  EXPECT_EQ(backend->load_calls(), 1);
  EXPECT_EQ(backend->batch_sizes(), std::vector<int>({3}));
}

TEST(InferenceRuntime, LargeRequestSplitsAcrossMultipleBackendRuns) {
  ThreadSafeQueue<PendingEval> request_queue;
  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue;
  auto backend = std::make_shared<RecordingBackend>();
  const FeatureEncoder encoder;

  InferenceRuntime runtime(
      InferenceChannels{
          .request_queue = &request_queue,
          .ready_queue = &ready_queue,
      },
      backend, &encoder,
      InferenceRuntimeOptions{
          .max_batch_size = 2,
          .flush_timeout = 0ms,
      });
  runtime.Start();

  ASSERT_TRUE(
      request_queue.push(MakePendingEval(/*task_id=*/7, /*num_items=*/5)));
  std::unique_ptr<TaggedGameTask> task = PopReadyTask(ready_queue);

  runtime.Stop();

  ASSERT_TRUE(task->response.has_value());
  EXPECT_EQ(task->task_id, 7);
  EXPECT_EQ(task->state, TaskState::kReady);
  ASSERT_EQ(task->response->items.size(), 5u);
  for (int item_id = 0; item_id < 5; ++item_id) {
    ExpectItemPayload(task->response->items[static_cast<std::size_t>(item_id)],
                      item_id);
  }
  EXPECT_EQ(backend->batch_sizes(), std::vector<int>({2, 2, 1}));
}

TEST(InferenceRuntime, SplitRequestCanShareLaterBatchWithNextRequest) {
  ThreadSafeQueue<PendingEval> request_queue;
  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue;
  auto backend = std::make_shared<RecordingBackend>();
  const FeatureEncoder encoder;

  InferenceRuntime runtime(
      InferenceChannels{
          .request_queue = &request_queue,
          .ready_queue = &ready_queue,
      },
      backend, &encoder,
      InferenceRuntimeOptions{
          .max_batch_size = 4,
          .flush_timeout = 5ms,
      });
  runtime.Start();

  ASSERT_TRUE(
      request_queue.push(MakePendingEval(/*task_id=*/10, /*num_items=*/5)));
  ASSERT_TRUE(
      request_queue.push(MakePendingEval(/*task_id=*/11, /*num_items=*/2)));

  std::unique_ptr<TaggedGameTask> first = PopReadyTask(ready_queue);
  std::unique_ptr<TaggedGameTask> second = PopReadyTask(ready_queue);

  runtime.Stop();

  ASSERT_TRUE(first->response.has_value());
  ASSERT_TRUE(second->response.has_value());
  EXPECT_EQ(first->task_id, 10);
  EXPECT_EQ(second->task_id, 11);
  ASSERT_EQ(first->response->items.size(), 5u);
  ASSERT_EQ(second->response->items.size(), 2u);
  for (int item_id = 0; item_id < 5; ++item_id) {
    ExpectItemPayload(first->response->items[static_cast<std::size_t>(item_id)],
                      item_id);
  }
  ExpectItemPayload(second->response->items[0], /*expected_item_id=*/5);
  ExpectItemPayload(second->response->items[1], /*expected_item_id=*/6);
  EXPECT_EQ(backend->batch_sizes(), std::vector<int>({4, 3}));
}

TEST(InferenceRuntime,
     RoutesAllItemsToActivePlayerBackendAcrossMultipleBackends) {
  ThreadSafeQueue<PendingEval> request_queue;
  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue;
  auto backend_zero = std::make_shared<MarkerBackend>(1000);
  auto backend_one = std::make_shared<MarkerBackend>(2000);
  const FeatureEncoder encoder;

  InferenceRuntime runtime(
      InferenceChannels{
          .request_queue = &request_queue,
          .ready_queue = &ready_queue,
      },
      std::vector<std::shared_ptr<InferenceBackend>>{
          backend_zero,
          backend_one,
      },
      &encoder,
      InferenceRuntimeOptions{
          .max_batch_size = 8,
          .flush_timeout = 5ms,
      });
  runtime.Start();

  ASSERT_TRUE(request_queue.push(MakeRoutedPendingEval(
      /*white_backend_slot=*/0, /*black_backend_slot=*/1,
      /*active_player_is_white=*/true, {false, true})));
  ASSERT_TRUE(request_queue.push(MakeRoutedPendingEval(
      /*white_backend_slot=*/1, /*black_backend_slot=*/0,
      /*active_player_is_white=*/true, {false, true})));

  std::unique_ptr<MatchTask> first = PopReadyMatchTask(ready_queue);
  std::unique_ptr<MatchTask> second = PopReadyMatchTask(ready_queue);

  runtime.Stop();

  ASSERT_TRUE(first->response.has_value());
  ASSERT_TRUE(second->response.has_value());
  EXPECT_EQ(first->state, TaskState::kReady);
  EXPECT_EQ(second->state, TaskState::kReady);
  ASSERT_EQ(first->response->items.size(), 2u);
  ASSERT_EQ(second->response->items.size(), 2u);
  ExpectMarker(first->response->items[0], 1000);
  ExpectMarker(first->response->items[1], 1001);
  ExpectMarker(second->response->items[0], 2000);
  ExpectMarker(second->response->items[1], 2001);
  EXPECT_EQ(backend_zero->load_calls(), 1);
  EXPECT_EQ(backend_one->load_calls(), 1);
  EXPECT_EQ(backend_zero->batch_sizes(), std::vector<int>({2}));
  EXPECT_EQ(backend_one->batch_sizes(), std::vector<int>({2}));
}

TEST(InferenceRuntime, MatchTaskRoutingFlipsAfterActivePlayerSwap) {
  MatchTask task;
  task.white_backend_slot = 3;
  task.black_backend_slot = 7;
  task.active_player_is_white = true;

  EvalRequestItem white_item;
  white_item.len = 1;
  white_item.positions[0] = WhiteToMovePosition();
  EvalRequestItem black_item;
  black_item.len = 1;
  black_item.positions[0] = BlackToMovePosition();

  EXPECT_EQ(task.BackendSlotForRequestItem(white_item), 3);
  EXPECT_EQ(task.BackendSlotForRequestItem(black_item), 3);

  task.active_player_is_white = false;
  EXPECT_EQ(task.BackendSlotForRequestItem(white_item), 7);
  EXPECT_EQ(task.BackendSlotForRequestItem(black_item), 7);
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
