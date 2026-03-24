#include "worker_runtime.h"

#include <chrono>
#include <future>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

namespace engine {
namespace {

class ImmediateSearcher final : public MCTSSearcher {
 public:
  SearchCoroutine Run() override {
    SearchResult result;
    result.game_result = lczero::GameResult::UNDECIDED;
    co_return result;
  }

  void CommitMove(lczero::Move /*move*/) override { ++commit_calls_; }

  int commit_calls() const { return commit_calls_; }

 private:
  int commit_calls_ = 0;
};

class YieldOnceSearcher final : public MCTSSearcher {
 public:
  SearchCoroutine Run() override {
    EvalRequest request;
    request.items.resize(1);
    request.items[0].len = 0;

    EvalResponse response = co_yield std::move(request);
    last_reply_size_ = response.items.size();

    SearchResult result;
    result.game_result = lczero::GameResult::UNDECIDED;
    result.legal_moves.resize(last_reply_size_);
    result.improved_policy.assign(last_reply_size_, 1.0f);
    co_return result;
  }

  void CommitMove(lczero::Move /*move*/) override { ++commit_calls_; }

  int commit_calls() const { return commit_calls_; }
  size_t last_reply_size() const { return last_reply_size_; }

 private:
  int commit_calls_ = 0;
  size_t last_reply_size_ = 0;
};

std::unique_ptr<GameTask> MakeTask(std::unique_ptr<MCTSSearcher> searcher) {
  auto task = std::make_unique<GameTask>();
  task->searcher = std::move(searcher);
  task->state = TaskState::kNew;
  return task;
}

WorkerChannels MakeChannels(
    ThreadSafeQueue<std::unique_ptr<GameTask>>& ready_queue,
    ThreadSafeQueue<PendingEval>& request_queue,
    ThreadSafeQueue<CompletedSearch>& completion_queue) {
  return WorkerChannels{
      .ready_queue = &ready_queue,
      .request_queue = &request_queue,
      .completion_queue = &completion_queue,
  };
}

class DerivedGameTask final : public GameTask {
 public:
  explicit DerivedGameTask(bool* destroyed_flag)
      : destroyed_flag_(destroyed_flag) {}
  ~DerivedGameTask() override {
    if (destroyed_flag_ != nullptr) *destroyed_flag_ = true;
  }

 private:
  bool* destroyed_flag_ = nullptr;
};

TEST(WorkerRuntime, NewTaskCompletesToCompletionQueue) {
  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue;
  ThreadSafeQueue<PendingEval> request_queue;
  ThreadSafeQueue<CompletedSearch> completion_queue;
  WorkerRuntime runtime(/*num_workers=*/1,
                        MakeChannels(ready_queue, request_queue,
                                     completion_queue));
  runtime.Start();

  auto searcher = std::make_unique<ImmediateSearcher>();
  ImmediateSearcher* searcher_raw = searcher.get();

  ASSERT_TRUE(ready_queue.push(MakeTask(std::move(searcher))));

  std::optional<CompletedSearch> completed = completion_queue.pop();
  ASSERT_TRUE(completed.has_value());
  ASSERT_TRUE(completed->task != nullptr);
  EXPECT_EQ(searcher_raw->commit_calls(), 0);

  request_queue.close();
  completion_queue.close();
  runtime.Stop();
}

TEST(WorkerRuntime, YieldEvalThenResponseThenComplete) {
  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue;
  ThreadSafeQueue<PendingEval> request_queue;
  ThreadSafeQueue<CompletedSearch> completion_queue;
  WorkerRuntime runtime(/*num_workers=*/1,
                        MakeChannels(ready_queue, request_queue,
                                     completion_queue));
  runtime.Start();

  auto searcher = std::make_unique<YieldOnceSearcher>();
  YieldOnceSearcher* searcher_raw = searcher.get();

  ASSERT_TRUE(ready_queue.push(MakeTask(std::move(searcher))));

  std::optional<PendingEval> pending = request_queue.pop();
  ASSERT_TRUE(pending.has_value());
  ASSERT_TRUE(pending->task != nullptr);
  EXPECT_EQ(pending->task->state, TaskState::kWaitingEval);
  ASSERT_EQ(pending->request.items.size(), 1u);

  EvalResponse response;
  response.items.resize(1);
  response.items[0].policy_logits = {0.1f, 0.2f};
  response.items[0].wdl_probs = {0.3f, 0.4f, 0.3f};
  pending->task->response = std::move(response);
  pending->task->state = TaskState::kReady;
  ASSERT_TRUE(ready_queue.push(std::move(pending->task)));

  std::optional<CompletedSearch> completed = completion_queue.pop();
  ASSERT_TRUE(completed.has_value());
  ASSERT_TRUE(completed->task != nullptr);
  EXPECT_EQ(searcher_raw->commit_calls(), 0);
  EXPECT_EQ(searcher_raw->last_reply_size(), 1u);
  EXPECT_EQ(completed->result.legal_moves.size(), 1u);
  EXPECT_EQ(completed->result.improved_policy.size(), 1u);

  request_queue.close();
  completion_queue.close();
  runtime.Stop();
}

TEST(WorkerRuntime, StopWakesBlockedWorkers) {
  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue;
  ThreadSafeQueue<PendingEval> request_queue;
  ThreadSafeQueue<CompletedSearch> completion_queue;
  WorkerRuntime runtime(/*num_workers=*/2,
                        MakeChannels(ready_queue, request_queue,
                                     completion_queue));
  runtime.Start();

  std::future<void> stop_future =
      std::async(std::launch::async, [&runtime] { runtime.Stop(); });
  EXPECT_EQ(stop_future.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  stop_future.get();

  request_queue.close();
  completion_queue.close();
}

TEST(WorkerRuntime, DerivedTaskSurvivesHandoffAndDeletesSafely) {
  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue;
  ThreadSafeQueue<PendingEval> request_queue;
  ThreadSafeQueue<CompletedSearch> completion_queue;
  WorkerRuntime runtime(/*num_workers=*/1,
                        MakeChannels(ready_queue, request_queue,
                                     completion_queue));
  runtime.Start();

  bool destroyed = false;
  auto searcher = std::make_unique<ImmediateSearcher>();
  auto task = std::make_unique<DerivedGameTask>(&destroyed);
  task->searcher = std::move(searcher);
  task->state = TaskState::kNew;

  ASSERT_TRUE(ready_queue.push(std::unique_ptr<GameTask>(std::move(task))));

  std::optional<CompletedSearch> completed = completion_queue.pop();
  ASSERT_TRUE(completed.has_value());
  ASSERT_TRUE(completed->task != nullptr);
  EXPECT_NE(dynamic_cast<DerivedGameTask*>(completed->task.get()), nullptr);
  completed->task.reset();
  EXPECT_TRUE(destroyed);

  request_queue.close();
  completion_queue.close();
  runtime.Stop();
}

TEST(WorkerRuntime, StartTwiceIsIgnored) {
  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue;
  ThreadSafeQueue<PendingEval> request_queue;
  ThreadSafeQueue<CompletedSearch> completion_queue;
  WorkerRuntime runtime(/*num_workers=*/1,
                        MakeChannels(ready_queue, request_queue,
                                     completion_queue));
  runtime.Start();
  runtime.Start();

  auto searcher = std::make_unique<ImmediateSearcher>();
  ASSERT_TRUE(ready_queue.push(MakeTask(std::move(searcher))));

  std::optional<CompletedSearch> completed = completion_queue.pop();
  ASSERT_TRUE(completed.has_value());

  request_queue.close();
  completion_queue.close();
  runtime.Stop();
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
