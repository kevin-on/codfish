#include "actor/mcts/match_coordinator.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "actor/backends/mock_backend.h"
#include "actor/mcts/runtime/match_types.h"

namespace engine {
namespace {

struct ScopedTempDir {
  ScopedTempDir() {
    path = std::filesystem::temp_directory_path() /
           ("codfish_match_coordinator_test_" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
  }

  ~ScopedTempDir() { std::filesystem::remove_all(path); }

  std::filesystem::path path;
};

struct ScopedCurrentPath {
  explicit ScopedCurrentPath(const std::filesystem::path& target)
      : previous(std::filesystem::current_path()) {
    std::filesystem::current_path(target);
  }

  ~ScopedCurrentPath() { std::filesystem::current_path(previous); }

  std::filesystem::path previous;
};

class ImmediateTerminalSearcher final : public MCTSSearcher {
 public:
  SearchCoroutine Run() override {
    SearchResult result;
    result.game_result = lczero::GameResult::DRAW;
    co_return result;
  }

  void CommitMove(lczero::Move /*move*/) override {}
};

class AlternatingMatchTaskFactory final : public GameTaskFactory {
 public:
  std::unique_ptr<GameTask> Create() override {
    auto task = std::make_unique<MatchTask>();
    task->searcher = std::make_unique<ImmediateTerminalSearcher>();
    task->state = TaskState::kNew;
    task->white_backend_slot = (next_task_index_ % 2 == 0) ? 0 : 1;
    task->black_backend_slot = 1 - task->white_backend_slot;
    ++next_task_index_;
    return task;
  }

 private:
  int next_task_index_ = 0;
};

using namespace std::chrono_literals;

TEST(MatchCoordinator, RunMatchWritesSinglePgnFileForPair) {
  ScopedTempDir temp_dir;
  auto backend_zero = std::make_shared<MockBackend>();
  auto backend_one = std::make_shared<MockBackend>();
  const FeatureEncoder encoder;

  MatchCoordinator coordinator(
      SearchCoordinatorConfig{
          .num_workers = 1,
          .inference =
              InferenceRuntimeOptions{
                  .max_batch_size = 4,
                  .flush_timeout = 0ms,
              },
      },
      std::make_unique<AlternatingMatchTaskFactory>(),
      {backend_zero, backend_one}, &encoder);

  const std::filesystem::path output_pgn = temp_dir.path / "match.pgn";
  coordinator.RunMatch(RunMatchOptions{
      .num_games = 2,
      .output_pgn_path = output_pgn,
      .player_names = {"player_a", "player_b"},
  });

  ASSERT_TRUE(std::filesystem::exists(output_pgn));
  const std::string pgn = [&output_pgn] {
    std::ifstream stream(output_pgn, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(stream),
                       std::istreambuf_iterator<char>());
  }();
  EXPECT_NE(pgn.find("[White \"player_a\"]"), std::string::npos);
  EXPECT_NE(pgn.find("[Black \"player_b\"]"), std::string::npos);
  EXPECT_NE(pgn.find("[White \"player_b\"]"), std::string::npos);
  EXPECT_NE(pgn.find("[Black \"player_a\"]"), std::string::npos);
  EXPECT_NE(pgn.find("[Result \"1/2-1/2\"]"), std::string::npos);
}

TEST(MatchCoordinator, RunMatchWritesBareRelativePgnPath) {
  ScopedTempDir temp_dir;
  ScopedCurrentPath scoped_cwd(temp_dir.path);
  auto backend_zero = std::make_shared<MockBackend>();
  auto backend_one = std::make_shared<MockBackend>();
  const FeatureEncoder encoder;

  MatchCoordinator coordinator(
      SearchCoordinatorConfig{
          .num_workers = 1,
          .inference =
              InferenceRuntimeOptions{
                  .max_batch_size = 4,
                  .flush_timeout = 0ms,
              },
      },
      std::make_unique<AlternatingMatchTaskFactory>(),
      {backend_zero, backend_one}, &encoder);

  const std::filesystem::path output_pgn = "match.pgn";
  coordinator.RunMatch(RunMatchOptions{
      .num_games = 2,
      .output_pgn_path = output_pgn,
      .player_names = {"player_a", "player_b"},
  });

  ASSERT_TRUE(std::filesystem::exists(temp_dir.path / output_pgn));
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
