#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>

#include "chunk_writer_runtime.h"
#include "engine/encoder.h"
#include "engine/infer/inference_backend.h"
#include "engine/infer/model_manifest.h"
#include "game_runner.h"
#include "inference_runtime.h"
#include "runtime_types.h"
#include "thread_safe_queue.h"
#include "worker_runtime.h"

namespace engine {

class GameTaskFactory {
 public:
  virtual ~GameTaskFactory() = default;
  virtual std::unique_ptr<GameTask> Create() = 0;
};

struct SearchCoordinatorOptions {
  int num_workers = 0;
  int num_initial_games = 0;
  std::optional<std::filesystem::path> raw_output_dir;
  uint64_t raw_chunk_max_bytes = ChunkWriterOptions::kDefaultMaxChunkBytes;
  InferenceRuntimeOptions inference;
};

class SearchCoordinator {
 public:
  SearchCoordinator(SearchCoordinatorOptions options,
                    std::unique_ptr<GameTaskFactory> task_factory,
                    std::shared_ptr<InferenceBackend> backend,
                    const FeatureEncoder* encoder, ModelManifest manifest);
  ~SearchCoordinator();

  SearchCoordinator(const SearchCoordinator&) = delete;
  SearchCoordinator& operator=(const SearchCoordinator&) = delete;

  void Start();
  void Stop();

 private:
  void SeedInitialTasks();

  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue_;
  ThreadSafeQueue<PendingEval> request_queue_;
  ThreadSafeQueue<CompletedSearch> completion_queue_;
  ThreadSafeQueue<CompletedGame> completed_game_queue_;

  SearchCoordinatorOptions options_;
  std::unique_ptr<GameTaskFactory> task_factory_;
  std::shared_ptr<InferenceBackend> backend_;
  const FeatureEncoder* encoder_ = nullptr;
  ModelManifest manifest_;

  WorkerRuntime worker_runtime_;
  InferenceRuntime inference_runtime_;
  GameRunner game_runner_;
  std::unique_ptr<ChunkWriterRuntime> chunk_writer_runtime_;
  bool started_ = false;
  bool stopped_ = false;
  mutable std::mutex state_mu_;
};

}  // namespace engine
