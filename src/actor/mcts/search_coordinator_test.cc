#include "search_coordinator.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "actor/backends/mock_backend.h"
#include "raw_chunk_format.h"

namespace engine {
namespace {

using namespace std::chrono_literals;

struct Probe {
  std::atomic<int> created{0};
  std::atomic<int> run_started{0};
  std::atomic<int> replies_seen{0};
  std::atomic<int> committed{0};
  std::atomic<int> destroyed{0};
};

struct ScopedTempDir {
  ScopedTempDir() {
    path = std::filesystem::temp_directory_path() /
           ("codfish_search_coordinator_test_" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
  }

  ~ScopedTempDir() { std::filesystem::remove_all(path); }

  std::filesystem::path path;
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

lczero::PositionHistory MakeStartHistory() {
  lczero::PositionHistory history;
  history.Reset(lczero::Position::FromFen(lczero::ChessBoard::kStartposFen));
  return history;
}

lczero::Move ParseStartMove(const char* uci) {
  const lczero::Position start =
      lczero::Position::FromFen(lczero::ChessBoard::kStartposFen);
  return start.GetBoard().ParseMove(uci);
}

std::vector<uint8_t> ReadFileBytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream.is_open()) {
    throw std::runtime_error("failed to open chunk file");
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream),
                              std::istreambuf_iterator<char>());
}

uint16_t ReadFirstSelectedMoveRaw(const std::filesystem::path& path) {
  const std::vector<uint8_t> bytes = ReadFileBytes(path);
  const raw_chunk_format::ParsedChunk chunk =
      raw_chunk_format::ParseChunk(bytes);
  if (chunk.version != raw_chunk_format::kChunkVersion) {
    throw std::runtime_error("unexpected chunk version");
  }
  if (chunk.records.empty() || chunk.records.front().plies.empty()) {
    throw std::runtime_error("missing ply data");
  }
  return chunk.records.front().plies.front().selected_move_raw;
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

class NonTerminalThenTerminalSearcher final : public MCTSSearcher {
 public:
  explicit NonTerminalThenTerminalSearcher(Probe* probe)
      : probe_(probe), selected_move_(ParseStartMove("e2e4")) {}
  ~NonTerminalThenTerminalSearcher() override { ++probe_->destroyed; }

  SearchCoroutine Run() override {
    ++probe_->run_started;
    SearchResult result;
    if (!committed_) {
      result.root_history = MakeStartHistory();
      result.selected_move = selected_move_;
      result.legal_moves = {selected_move_};
      result.improved_policy = {1.0f};
      co_return result;
    }

    result.game_result = lczero::GameResult::DRAW;
    co_return result;
  }

  void CommitMove(lczero::Move move) override {
    assert(!committed_);
    assert(move == selected_move_);
    committed_ = true;
    ++probe_->committed;
  }

 private:
  Probe* probe_ = nullptr;
  lczero::Move selected_move_;
  bool committed_ = false;
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

TEST(SearchCoordinator, RawOutputDirStartsWriterAndPersistsCompletedGame) {
  Probe probe;
  ScopedTempDir temp_dir;
  auto backend = std::make_shared<MockBackend>();
  const FeatureEncoder encoder;
  const lczero::Move expected_move = ParseStartMove("e2e4");

  SearchCoordinator coordinator(
      SearchCoordinatorOptions{
          .num_workers = 1,
          .num_initial_games = 1,
          .raw_output_dir = temp_dir.path,
          .inference =
              InferenceRuntimeOptions{
                  .max_batch_size = 1,
                  .flush_timeout = 0ms,
              },
      },
      std::make_unique<CountingTaskFactory<NonTerminalThenTerminalSearcher>>(
          &probe),
      backend, &encoder, ModelManifest{});

  coordinator.Start();
  ASSERT_TRUE(WaitUntil([&probe] { return probe.destroyed.load() == 1; }));
  coordinator.Stop();

  EXPECT_EQ(probe.created.load(), 1);
  EXPECT_EQ(probe.run_started.load(), 2);
  EXPECT_EQ(probe.committed.load(), 1);
  EXPECT_EQ(probe.destroyed.load(), 1);

  const std::filesystem::path chunk_path =
      temp_dir.path / raw_chunk_format::ChunkFileName(1);
  ASSERT_TRUE(std::filesystem::exists(chunk_path));
  EXPECT_EQ(ReadFirstSelectedMoveRaw(chunk_path), expected_move.raw_data());
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
