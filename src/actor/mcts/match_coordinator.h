#pragma once

#include <array>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "actor/mcts/runtime/inference_runtime.h"
#include "actor/mcts/runtime/match_game_runner.h"
#include "actor/mcts/search_coordinator.h"

namespace engine {

struct RunMatchOptions {
  int num_games = 0;
  std::filesystem::path output_pgn_path;
  std::array<std::string, 2> player_names;
};

class MatchCoordinator {
 public:
  MatchCoordinator(SearchCoordinatorConfig config,
                   std::unique_ptr<GameTaskFactory> task_factory,
                   std::array<std::shared_ptr<InferenceBackend>, 2> backends,
                   const FeatureEncoder* encoder);
  ~MatchCoordinator();

  MatchCoordinator(const MatchCoordinator&) = delete;
  MatchCoordinator& operator=(const MatchCoordinator&) = delete;

  void RunMatch(const RunMatchOptions& options);

 private:
  void StartRun(const RunMatchOptions& options);
  void StopRun();
  void WaitForCompletedGames(int num_games);
  void OnCompletedGame(CompletedMatchGame completed_game);
  void SeedGameTasks(int num_games);
  void WritePgn(const RunMatchOptions& options) const;

  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue_;
  ThreadSafeQueue<PendingEval> request_queue_;
  ThreadSafeQueue<CompletedSearch> completion_queue_;

  SearchCoordinatorConfig config_;
  std::unique_ptr<GameTaskFactory> task_factory_;
  std::array<std::shared_ptr<InferenceBackend>, 2> backends_;
  const FeatureEncoder* encoder_ = nullptr;

  WorkerRuntime worker_runtime_;
  InferenceRuntime inference_runtime_;
  MatchGameRunner game_runner_;
  std::condition_variable completion_cv_;
  std::vector<CompletedMatchGame> completed_games_;
  int completed_games_count_ = 0;
  bool started_ = false;
  bool stopped_ = false;
  mutable std::mutex state_mu_;
};

}  // namespace engine
