#include "search_coordinator.h"

#include <cassert>
#include <utility>

namespace engine {

SearchCoordinator::SearchCoordinator(SearchCoordinatorOptions options,
                                     std::unique_ptr<GameTaskFactory> task_factory,
                                     std::shared_ptr<InferenceBackend> backend,
                                     const FeatureEncoder* encoder,
                                     ModelManifest manifest)
    : options_(options),
      task_factory_(std::move(task_factory)),
      backend_(std::move(backend)),
      encoder_(encoder),
      manifest_(manifest),
      worker_runtime_(
          options_.num_workers,
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
          backend_, encoder_, manifest_, options_.inference),
      game_runner_(GameRunnerChannels{
          .completion_queue = &completion_queue_,
          .ready_queue = &ready_queue_,
          .completed_game_queue = &completed_game_queue_,
      }) {
  assert(options_.num_workers > 0);
  assert(options_.num_initial_games >= 0);
  assert(task_factory_ != nullptr);
  assert(backend_ != nullptr);
  assert(encoder_ != nullptr);
}

SearchCoordinator::~SearchCoordinator() { Stop(); }

void SearchCoordinator::Start() {
  std::lock_guard<std::mutex> lock(state_mu_);
  if (started_) return;
  if (stopped_) return;

  game_runner_.Start();
  inference_runtime_.Start();
  worker_runtime_.Start();
  SeedInitialTasks();
  started_ = true;
}

void SearchCoordinator::Stop() {
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
  completed_game_queue_.close();
}

void SearchCoordinator::SeedInitialTasks() {
  for (int i = 0; i < options_.num_initial_games; ++i) {
    std::unique_ptr<GameTask> task = task_factory_->Create();
    assert(task != nullptr);
    assert(task->searcher != nullptr);
    assert(task->state == TaskState::kNew);
    const bool pushed = ready_queue_.push(std::move(task));
    assert(pushed);
  }
}

}  // namespace engine
