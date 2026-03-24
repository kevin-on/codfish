#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "searcher.h"
#include "thread_safe_queue.h"

namespace engine {

enum class TaskState : uint8_t {
  kNew,
  kWaitingEval,
  kReady,
};

struct GameTask {
  virtual ~GameTask() = default;

  std::unique_ptr<MCTSSearcher> searcher;
  std::optional<SearchCoroutine> coroutine;
  TaskState state = TaskState::kNew;
  std::optional<EvalResponse> response;
};

struct PendingEval {
  std::unique_ptr<GameTask> task;
  EvalRequest request;
};

struct CompletedSearch {
  std::unique_ptr<GameTask> task;
  SearchResult result;
};

struct WorkerChannels {
  ThreadSafeQueue<std::unique_ptr<GameTask>>* ready_queue = nullptr;
  ThreadSafeQueue<PendingEval>* request_queue = nullptr;
  ThreadSafeQueue<CompletedSearch>* completion_queue = nullptr;
};

class WorkerRuntime {
 public:
  WorkerRuntime(int num_workers, WorkerChannels channels);
  ~WorkerRuntime();

  WorkerRuntime(const WorkerRuntime&) = delete;
  WorkerRuntime& operator=(const WorkerRuntime&) = delete;

  void Start();
  void Stop();

 private:
  void WorkerLoop();

  int num_workers_ = 0;
  WorkerChannels channels_;
  bool started_ = false;
  bool stopped_ = false;
  mutable std::mutex state_mu_;
  std::vector<std::thread> workers_;
};

}  // namespace engine
