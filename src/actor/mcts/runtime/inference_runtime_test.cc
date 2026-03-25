#include "actor/mcts/runtime/inference_runtime.h"

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

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
  Status Load(const ModelManifest& manifest) override {
    policy_size_ = manifest.policy_size;
    ++load_calls_;
    return Status::Ok();
  }

  Status Run(const InferenceBatch& batch, InferenceOutputs* out) override {
    if (out == nullptr) return Status::Error("output buffer is null");
    if (policy_size_ <= 0) return Status::Error("backend not loaded");

    batch_sizes_.push_back(batch.batch_size);
    out->policy_logits.clear();
    out->wdl_probs.clear();

    for (int batch_idx = 0; batch_idx < batch.batch_size; ++batch_idx) {
      const int item_id = next_item_id_++;
      for (int policy_idx = 0; policy_idx < policy_size_; ++policy_idx) {
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
  int policy_size_ = 0;
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

void ExpectItemPayload(const EvalResponseItem& item, int expected_item_id,
                       int policy_size) {
  ASSERT_EQ(item.policy_logits.size(), static_cast<std::size_t>(policy_size));
  ASSERT_EQ(item.wdl_probs.size(), 3u);
  for (int policy_idx = 0; policy_idx < policy_size; ++policy_idx) {
    EXPECT_FLOAT_EQ(item.policy_logits[policy_idx],
                    static_cast<float>(expected_item_id * 10 + policy_idx));
  }
  EXPECT_FLOAT_EQ(item.wdl_probs[0], static_cast<float>(expected_item_id));
  EXPECT_FLOAT_EQ(item.wdl_probs[1],
                  static_cast<float>(expected_item_id + 100));
  EXPECT_FLOAT_EQ(item.wdl_probs[2],
                  static_cast<float>(expected_item_id + 200));
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
      backend, &encoder, ModelManifest{.policy_size = 4},
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
  ExpectItemPayload(first->response->items[0], /*expected_item_id=*/0,
                    /*policy_size=*/4);
  ExpectItemPayload(second->response->items[0], /*expected_item_id=*/1,
                    /*policy_size=*/4);
  ExpectItemPayload(second->response->items[1], /*expected_item_id=*/2,
                    /*policy_size=*/4);
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
      backend, &encoder, ModelManifest{.policy_size = 3},
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
                      item_id, /*policy_size=*/3);
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
      backend, &encoder, ModelManifest{.policy_size = 2},
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
                      item_id, /*policy_size=*/2);
  }
  ExpectItemPayload(second->response->items[0], /*expected_item_id=*/5,
                    /*policy_size=*/2);
  ExpectItemPayload(second->response->items[1], /*expected_item_id=*/6,
                    /*policy_size=*/2);
  EXPECT_EQ(backend->batch_sizes(), std::vector<int>({4, 3}));
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
