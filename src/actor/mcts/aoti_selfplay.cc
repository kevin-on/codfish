#include "actor/mcts/aoti_selfplay.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "actor/backends/aoti_backend.h"
#include "actor/mcts/output/raw_chunk_format.h"
#include "actor/mcts/search_coordinator.h"
#include "actor/mcts/searchers/gumbel/gumbel_mcts.h"

namespace engine {
namespace {

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

void ValidateOptions(const AotiSelfPlayOptions& options) {
  if (options.model_package_path.empty()) {
    throw std::runtime_error("model_package_path must be provided");
  }
  if (options.input_channels <= 0) {
    throw std::runtime_error("input_channels must be positive");
  }
  if (options.policy_size <= 0) {
    throw std::runtime_error("policy_size must be positive");
  }
  if (options.raw_output_dir.empty()) {
    throw std::runtime_error("raw_output_dir must be provided");
  }
  if (options.num_workers <= 0) {
    throw std::runtime_error("num_workers must be positive");
  }
  if (options.num_games <= 0) {
    throw std::runtime_error("num_games must be positive");
  }
  if (options.raw_chunk_max_bytes == 0) {
    throw std::runtime_error("raw_chunk_max_bytes must be positive");
  }
  if (options.num_action <= 0) {
    throw std::runtime_error("num_action must be positive");
  }
  if (options.num_simulation <= 0) {
    throw std::runtime_error("num_simulation must be positive");
  }
  if (options.c_puct < 0.0f) {
    throw std::runtime_error("c_puct must be non-negative");
  }
  if (options.c_visit < 0.0f) {
    throw std::runtime_error("c_visit must be non-negative");
  }
  if (options.c_scale < 0.0f) {
    throw std::runtime_error("c_scale must be non-negative");
  }
}

std::vector<std::filesystem::path> ChunkPathsInDir(
    const std::filesystem::path& output_dir) {
  std::vector<std::filesystem::path> paths;
  if (!std::filesystem::exists(output_dir)) {
    return paths;
  }
  for (const std::filesystem::directory_entry& entry :
       std::filesystem::directory_iterator(output_dir)) {
    if (!entry.is_regular_file()) continue;
    if (entry.path().extension() != ".bin") continue;
    paths.push_back(entry.path());
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

std::vector<uint8_t> ReadFileBytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream.is_open()) {
    throw std::runtime_error("failed to open chunk file");
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream),
                              std::istreambuf_iterator<char>());
}

void ValidateRawChunkOutput(const std::filesystem::path& output_dir,
                            int expected_games) {
  int total = 0;
  for (const std::filesystem::path& path : ChunkPathsInDir(output_dir)) {
    const raw_chunk_format::ParsedChunk chunk =
        raw_chunk_format::ParseChunk(ReadFileBytes(path));
    for (const raw_chunk_format::StoredRawGame& record : chunk.records) {
      if (record.game_result == lczero::GameResult::UNDECIDED) {
        throw std::runtime_error("chunk contains undecided game");
      }
    }
    total += static_cast<int>(chunk.records.size());
  }
  if (total != expected_games) {
    throw std::runtime_error("expected completed game count does not match");
  }
}

}  // namespace

void RunAotiSelfPlay(const AotiSelfPlayOptions& options) {
  ValidateOptions(options);

  const FeatureEncoder encoder;
  const GumbelMCTSConfig search_config{
      .num_action = options.num_action,
      .num_simulation = options.num_simulation,
      .c_puct = options.c_puct,
      .c_visit = options.c_visit,
      .c_scale = options.c_scale,
  };

  auto backend = std::make_shared<AotiBackend>(
      options.model_package_path, options.input_channels, options.policy_size);
  SearchCoordinator coordinator(
      SearchCoordinatorConfig{
          .num_workers = options.num_workers,
          .inference =
              InferenceRuntimeOptions{
                  .max_batch_size = options.num_action,
                  .flush_timeout = std::chrono::microseconds{0},
              },
      },
      std::make_unique<GumbelTaskFactory>(search_config), backend, &encoder);

  coordinator.RunGames(RunGamesOptions{
      .num_games = options.num_games,
      .raw_output_dir = options.raw_output_dir,
      .raw_chunk_max_bytes = options.raw_chunk_max_bytes,
  });

  ValidateRawChunkOutput(options.raw_output_dir, options.num_games);
}

}  // namespace engine
