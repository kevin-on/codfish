#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "actor/backends/mock_backend.h"
#include "actor/mcts/output/raw_chunk_format.h"
#include "actor/mcts/search_coordinator.h"
#include "actor/mcts/searchers/gumbel/gumbel_mcts.h"

namespace engine {
namespace {

struct ScopedTempDir {
  ScopedTempDir() {
    path = std::filesystem::temp_directory_path() /
           ("codfish_search_coordinator_smoke_test_" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
  }

  ~ScopedTempDir() { std::filesystem::remove_all(path); }

  std::filesystem::path path;
};

using namespace std::chrono_literals;

std::vector<uint8_t> ReadFileBytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream.is_open()) {
    throw std::runtime_error("failed to open chunk file");
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream),
                              std::istreambuf_iterator<char>());
}

bool SelectedMoveIsInPolicy(const raw_chunk_format::StoredPly& ply) {
  for (const raw_chunk_format::StoredPolicyEntry& entry : ply.policy) {
    if (entry.move_uci == ply.selected_move_uci) return true;
  }
  return false;
}

class GumbelTaskFactory final : public GameTaskFactory {
 public:
  explicit GumbelTaskFactory(GumbelMCTSConfig config) : config_(config) {}

  std::unique_ptr<GameTask> Create() override {
    auto task = std::make_unique<GameTask>();
    task->searcher = std::make_unique<GumbelMCTS>(config_);
    task->state = TaskState::kNew;
    return task;
  }

 private:
  GumbelMCTSConfig config_;
};

TEST(SearchCoordinatorSmoke,
     GumbelMctsWithMockBackendCompletesGameAndWritesChunk) {
  ScopedTempDir temp_dir;
  auto backend = std::make_shared<MockBackend>();
  const FeatureEncoder encoder;

  // Keep the search non-trivial enough to exercise batched evals and
  // sequential halving, but small enough that a full mock self-play game
  // remains practical as a smoke test.
  constexpr GumbelMCTSConfig kSearchConfig{
      .num_action = 8,
      .num_simulation = 16,
      .c_puct = 1.0f,
      .c_visit = 1.0f,
      .c_scale = 1.0f,
  };

  SearchCoordinator coordinator(
      SearchCoordinatorConfig{
          .num_workers = 1,
          .inference =
              InferenceRuntimeOptions{
                  .max_batch_size = kSearchConfig.num_action,
                  .flush_timeout = 0ms,
              },
      },
      std::make_unique<GumbelTaskFactory>(kSearchConfig), backend, &encoder);

  const std::filesystem::path chunk_path =
      temp_dir.path / raw_chunk_format::ChunkFileName(1);
  const std::string expected_initial_fen = lczero::PositionToFen(
      lczero::Position::FromFen(lczero::ChessBoard::kStartposFen));

  coordinator.RunGames(RunGamesOptions{
      .num_games = 1,
      .raw_output_dir = temp_dir.path,
  });

  const raw_chunk_format::ParsedChunk parsed_chunk =
      raw_chunk_format::ParseChunk(ReadFileBytes(chunk_path));
  EXPECT_EQ(parsed_chunk.version, raw_chunk_format::kChunkVersion);
  ASSERT_EQ(parsed_chunk.records.size(), 1u);

  const raw_chunk_format::StoredRawGame& record = parsed_chunk.records.front();
  ASSERT_TRUE(record.initial_fen.has_value());
  EXPECT_EQ(*record.initial_fen, expected_initial_fen);
  EXPECT_NE(record.game_result, lczero::GameResult::UNDECIDED);
  ASSERT_FALSE(record.plies.empty());

  for (const raw_chunk_format::StoredPly& ply : record.plies) {
    ASSERT_FALSE(ply.policy.empty());
    EXPECT_TRUE(SelectedMoveIsInPolicy(ply));

    float prob_sum = 0.0f;
    for (const raw_chunk_format::StoredPolicyEntry& entry : ply.policy) {
      EXPECT_GE(entry.prob, 0.0f);
      prob_sum += entry.prob;
    }
    EXPECT_NEAR(prob_sum, 1.0f, 1e-4f);
  }
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
