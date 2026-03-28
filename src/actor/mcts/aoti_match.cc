#include "actor/mcts/aoti_match.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include "actor/backends/aoti_backend.h"
#include "actor/mcts/match_coordinator.h"
#include "actor/mcts/runtime/match_types.h"
#include "actor/mcts/search_coordinator.h"
#include "actor/mcts/searchers/gumbel/gumbel_mcts.h"

namespace engine {
namespace {

class AlternatingGumbelMatchTaskFactory final : public GameTaskFactory {
 public:
  explicit AlternatingGumbelMatchTaskFactory(GumbelMCTSConfig config)
      : config_(config) {}

  std::unique_ptr<GameTask> Create() override {
    auto task = std::make_unique<MatchTask>();
    task->searcher = std::make_unique<GumbelMCTS>(config_);
    task->state = TaskState::kNew;
    task->white_backend_slot = (next_game_index_ % 2 == 0) ? 0 : 1;
    task->black_backend_slot = 1 - task->white_backend_slot;
    ++next_game_index_;
    return task;
  }

 private:
  GumbelMCTSConfig config_;
  int next_game_index_ = 0;
};

void ValidateOptions(const AotiMatchOptions& options) {
  for (int slot = 0; slot < 2; ++slot) {
    if (options.model_package_paths[slot].empty()) {
      throw std::runtime_error("model_package_paths must be provided");
    }
    if (options.player_names[slot].empty()) {
      throw std::runtime_error("player_names must be provided");
    }
  }
  if (options.input_channels <= 0) {
    throw std::runtime_error("input_channels must be positive");
  }
  if (options.policy_size <= 0) {
    throw std::runtime_error("policy_size must be positive");
  }
  if (options.output_pgn_path.empty()) {
    throw std::runtime_error("output_pgn_path must be provided");
  }
  if (options.num_workers <= 0) {
    throw std::runtime_error("num_workers must be positive");
  }
  if (options.num_games <= 0) {
    throw std::runtime_error("num_games must be positive");
  }
  if ((options.num_games % 2) != 0) {
    throw std::runtime_error("num_games must be even");
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

}  // namespace

void RunAotiMatch(const AotiMatchOptions& options) {
  ValidateOptions(options);

  const FeatureEncoder encoder;
  const GumbelMCTSConfig search_config{
      .num_action = options.num_action,
      .num_simulation = options.num_simulation,
      .c_puct = options.c_puct,
      .c_visit = options.c_visit,
      .c_scale = options.c_scale,
  };

  std::array<std::shared_ptr<InferenceBackend>, 2> backends{
      std::make_shared<AotiBackend>(options.model_package_paths[0],
                                    options.input_channels,
                                    options.policy_size),
      std::make_shared<AotiBackend>(options.model_package_paths[1],
                                    options.input_channels,
                                    options.policy_size),
  };
  MatchCoordinator coordinator(
      SearchCoordinatorConfig{
          .num_workers = options.num_workers,
          .inference =
              InferenceRuntimeOptions{
                  .max_batch_size = options.num_action,
                  .flush_timeout = std::chrono::microseconds{0},
              },
      },
      std::make_unique<AlternatingGumbelMatchTaskFactory>(search_config),
      std::move(backends), &encoder);

  coordinator.RunMatch(RunMatchOptions{
      .num_games = options.num_games,
      .output_pgn_path = options.output_pgn_path,
      .player_names = options.player_names,
  });
}

}  // namespace engine
