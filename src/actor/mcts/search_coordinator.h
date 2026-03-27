#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>

#include "actor/mcts/output/chunk_writer_runtime.h"
#include "actor/mcts/primitives/thread_safe_queue.h"
#include "actor/mcts/runtime/game_runner.h"
#include "actor/mcts/runtime/inference_runtime.h"
#include "actor/mcts/runtime/task_types.h"
#include "actor/mcts/runtime/worker_runtime.h"
#include "engine/encoder.h"
#include "engine/infer/inference_backend.h"
#include "engine/infer/model_manifest.h"

namespace engine {

class GameTaskFactory {
 public:
  virtual ~GameTaskFactory() = default;
  virtual std::unique_ptr<GameTask> Create() = 0;
};

struct SearchCoordinatorConfig {
  int num_workers = 0;
  InferenceRuntimeOptions inference;
};

struct RunGamesOptions {
  int num_games = 0;
  std::optional<std::filesystem::path> raw_output_dir;
  uint64_t raw_chunk_max_bytes = ChunkWriterOptions::kDefaultMaxChunkBytes;
  std::chrono::steady_clock::duration timeout;
};

class SearchCoordinator {
 public:
  SearchCoordinator(SearchCoordinatorConfig config,
                    std::unique_ptr<GameTaskFactory> task_factory,
                    std::shared_ptr<InferenceBackend> backend,
                    const FeatureEncoder* encoder, ModelManifest manifest);
  ~SearchCoordinator();

  SearchCoordinator(const SearchCoordinator&) = delete;
  SearchCoordinator& operator=(const SearchCoordinator&) = delete;

  bool RunGames(const RunGamesOptions& options);

 private:
  void StartRun(const RunGamesOptions& options);
  void StopRun();
  bool WaitForCompletedGames(const RunGamesOptions& options);
  void OnCompletedGame();
  void SeedGameTasks(int num_games);

  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue_;
  ThreadSafeQueue<PendingEval> request_queue_;
  ThreadSafeQueue<CompletedSearch> completion_queue_;
  ThreadSafeQueue<CompletedGame> completed_game_queue_;

  SearchCoordinatorConfig config_;
  std::unique_ptr<GameTaskFactory> task_factory_;
  std::shared_ptr<InferenceBackend> backend_;
  const FeatureEncoder* encoder_ = nullptr;
  ModelManifest manifest_;

  WorkerRuntime worker_runtime_;
  InferenceRuntime inference_runtime_;
  GameRunner game_runner_;
  std::unique_ptr<ChunkWriterRuntime> chunk_writer_runtime_;
  std::condition_variable completion_cv_;
  int completed_games_ = 0;
  bool started_ = false;
  bool stopped_ = false;
  mutable std::mutex state_mu_;
};

}  // namespace engine
