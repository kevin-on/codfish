#include "search_coordinator.h"

#include <cassert>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "actor/backends/mock_backend.h"

namespace engine {
namespace {

using namespace std::chrono_literals;

struct Probe {
  std::atomic<int> created{0};
  std::atomic<int> run_started{0};
  std::atomic<int> replies_seen{0};
  std::atomic<int> destroyed{0};
};

bool WaitUntil(const std::function<bool()>& predicate,
               std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) return true;
    std::this_thread::sleep_for(10ms);
  }
  return predicate();
}

class ImmediateTerminalSearcher final : public MCTSSearcher {
 public:
  explicit ImmediateTerminalSearcher(Probe* probe) : probe_(probe) {}
  ~ImmediateTerminalSearcher() override { ++probe_->destroyed; }

  SearchCoroutine Run() override {
    ++probe_->run_started;
    SearchResult result;
    result.game_result = lczero::GameResult::DRAW;
    co_return result;
  }

  void CommitMove(lczero::Move /*move*/) override {}

 private:
  Probe* probe_ = nullptr;
};

class YieldOnceTerminalSearcher final : public MCTSSearcher {
 public:
  explicit YieldOnceTerminalSearcher(Probe* probe) : probe_(probe) {}
  ~YieldOnceTerminalSearcher() override { ++probe_->destroyed; }

  SearchCoroutine Run() override {
    ++probe_->run_started;

    EvalRequest request;
    request.items.resize(1);
    request.items[0].len = 1;
    request.items[0].positions[0] = lczero::Position();

    EvalResponse response = co_yield std::move(request);
    assert(response.items.size() == 1);
    ++probe_->replies_seen;

    SearchResult result;
    result.game_result = lczero::GameResult::DRAW;
    co_return result;
  }

  void CommitMove(lczero::Move /*move*/) override {}

 private:
  Probe* probe_ = nullptr;
};

template <typename Searcher>
class CountingTaskFactory final : public GameTaskFactory {
 public:
  explicit CountingTaskFactory(Probe* probe) : probe_(probe) {}

  std::unique_ptr<GameTask> Create() override {
    ++probe_->created;
    auto task = std::make_unique<GameTask>();
    task->searcher = std::make_unique<Searcher>(probe_);
    task->state = TaskState::kNew;
    return task;
  }

 private:
  Probe* probe_ = nullptr;
};

TEST(SearchCoordinator, StartSeedsInitialGamesAndRunsTerminalTasks) {
  Probe probe;
  auto backend = std::make_shared<MockBackend>();
  const FeatureEncoder encoder;

  SearchCoordinator coordinator(
      SearchCoordinatorOptions{
          .num_workers = 2,
          .num_initial_games = 3,
          .inference =
              InferenceRuntimeOptions{
                  .max_batch_size = 4,
                  .flush_timeout = 0ms,
              },
      },
      std::make_unique<CountingTaskFactory<ImmediateTerminalSearcher>>(&probe),
      backend, &encoder, ModelManifest{});

  coordinator.Start();
  ASSERT_TRUE(WaitUntil([&probe] { return probe.destroyed.load() == 3; }));
  coordinator.Stop();

  EXPECT_EQ(probe.created.load(), 3);
  EXPECT_EQ(probe.run_started.load(), 3);
  EXPECT_EQ(probe.replies_seen.load(), 0);
  EXPECT_EQ(probe.destroyed.load(), 3);
}

TEST(SearchCoordinator, StartWiresInferencePathForYieldingTasks) {
  Probe probe;
  auto backend = std::make_shared<MockBackend>();
  const FeatureEncoder encoder;

  SearchCoordinator coordinator(
      SearchCoordinatorOptions{
          .num_workers = 1,
          .num_initial_games = 2,
          .inference =
              InferenceRuntimeOptions{
                  .max_batch_size = 2,
                  .flush_timeout = 0ms,
              },
      },
      std::make_unique<CountingTaskFactory<YieldOnceTerminalSearcher>>(&probe),
      backend, &encoder, ModelManifest{});

  coordinator.Start();
  ASSERT_TRUE(WaitUntil([&probe] { return probe.destroyed.load() == 2; }));
  coordinator.Stop();

  EXPECT_EQ(probe.created.load(), 2);
  EXPECT_EQ(probe.run_started.load(), 2);
  EXPECT_EQ(probe.replies_seen.load(), 2);
  EXPECT_EQ(probe.destroyed.load(), 2);
}

TEST(SearchCoordinator, StopIsSafeWhenStartedIdle) {
  Probe probe;
  auto backend = std::make_shared<MockBackend>();
  const FeatureEncoder encoder;

  SearchCoordinator coordinator(
      SearchCoordinatorOptions{
          .num_workers = 1,
          .num_initial_games = 0,
          .inference =
              InferenceRuntimeOptions{
                  .max_batch_size = 1,
                  .flush_timeout = 0ms,
              },
      },
      std::make_unique<CountingTaskFactory<ImmediateTerminalSearcher>>(&probe),
      backend, &encoder, ModelManifest{});

  coordinator.Start();
  coordinator.Stop();

  EXPECT_EQ(probe.created.load(), 0);
  EXPECT_EQ(probe.run_started.load(), 0);
  EXPECT_EQ(probe.replies_seen.load(), 0);
  EXPECT_EQ(probe.destroyed.load(), 0);
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
