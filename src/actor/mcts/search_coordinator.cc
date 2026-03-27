#include "actor/mcts/search_coordinator.h"

#include <cassert>
#include <stdexcept>
#include <utility>

namespace engine {

SearchCoordinator::SearchCoordinator(
    SearchCoordinatorConfig config,
    std::unique_ptr<GameTaskFactory> task_factory,
    std::shared_ptr<InferenceBackend> backend, const FeatureEncoder* encoder,
    ModelManifest manifest)
    : config_(config),
      task_factory_(std::move(task_factory)),
      backend_(std::move(backend)),
      encoder_(encoder),
      manifest_(manifest),
      worker_runtime_(config_.num_workers,
                      WorkerChannels{
                          .ready_queue = &ready_queue_,
                          .request_queue = &request_queue_,
                          .completion_queue = &completion_queue_,
                      }),
      inference_runtime_(
          InferenceChannels{
              .request_queue = &request_queue_,
              .ready_queue = &ready_queue_,
          },
          backend_, encoder_, manifest_, config_.inference),
      game_runner_(GameRunnerChannels{
          .completion_queue = &completion_queue_,
          .ready_queue = &ready_queue_,
          .completed_game_queue = &completed_game_queue_,
          .on_completed_game = [this] { OnCompletedGame(); },
      }) {
  assert(config_.num_workers > 0);
  assert(task_factory_ != nullptr);
  assert(backend_ != nullptr);
  assert(encoder_ != nullptr);
}

SearchCoordinator::~SearchCoordinator() { StopRun(); }

void SearchCoordinator::RunGames(const RunGamesOptions& options) {
  bool cleanup_needed = true;
  try {
    StartRun(options);
    WaitForCompletedGames(options.num_games);
    StopRun();
    cleanup_needed = false;
  } catch (...) {
    if (cleanup_needed) {
      StopRun();
    }
    throw;
  }
}

void SearchCoordinator::StartRun(const RunGamesOptions& options) {
  assert(options.num_games >= 0);
  assert(options.raw_chunk_max_bytes > 0);

  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (started_ || stopped_) {
      throw std::logic_error(
          "SearchCoordinator::RunGames() cannot be called more than once");
    }
    completed_games_ = 0;
    started_ = true;
  }

  if (options.raw_output_dir.has_value()) {
    chunk_writer_runtime_ = std::make_unique<ChunkWriterRuntime>(
        ChunkWriterChannels{
            .completed_game_queue = &completed_game_queue_,
        },
        ChunkWriterOptions{
            .output_dir = *options.raw_output_dir,
            .max_chunk_bytes = options.raw_chunk_max_bytes,
        });
  }
  if (chunk_writer_runtime_ != nullptr) {
    chunk_writer_runtime_->Start();
  }
  game_runner_.Start();
  inference_runtime_.Start();
  worker_runtime_.Start();
  SeedGameTasks(options.num_games);
}

void SearchCoordinator::StopRun() {
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (!started_ || stopped_) return;
    stopped_ = true;
  }

  // TODO: Current shutdown is not a full lifecycle drain. In-flight inference
  // and non-terminal game progression can still race with queue closure and
  // need a dedicated Stop() contract in a follow-up change.
  worker_runtime_.Stop();
  inference_runtime_.Stop();
  game_runner_.Stop();
  if (chunk_writer_runtime_ != nullptr) {
    chunk_writer_runtime_->Stop();
    chunk_writer_runtime_.reset();
  } else {
    completed_game_queue_.close();
  }
}

void SearchCoordinator::WaitForCompletedGames(int num_games) {
  std::unique_lock<std::mutex> lock(state_mu_);
  if (num_games == 0) return;

  completion_cv_.wait(lock,
                      [this, num_games] { return completed_games_ >= num_games; });
}

void SearchCoordinator::OnCompletedGame() {
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    ++completed_games_;
  }
  completion_cv_.notify_all();
}

void SearchCoordinator::SeedGameTasks(int num_games) {
  for (int i = 0; i < num_games; ++i) {
    std::unique_ptr<GameTask> task = task_factory_->Create();
    assert(task != nullptr);
    assert(task->searcher != nullptr);
    assert(task->state == TaskState::kNew);
    const bool pushed = ready_queue_.push(std::move(task));
    if (!pushed) {
      throw std::runtime_error("ready queue rejected seeded game task");
    }
  }
}

}  // namespace engine
