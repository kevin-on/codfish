#pragma once

#include <mutex>
#include <thread>

#include "actor/mcts/primitives/thread_safe_queue.h"
#include "actor/mcts/runtime/task_types.h"

namespace engine {

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
