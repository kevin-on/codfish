#pragma once

#include <memory>
#include <mutex>
#include <thread>

#include "actor/mcts/primitives/thread_safe_queue.h"
#include "actor/mcts/runtime/task_types.h"

namespace engine {

struct GameRunnerChannels {
  ThreadSafeQueue<CompletedSearch>* completion_queue = nullptr;
  ThreadSafeQueue<std::unique_ptr<GameTask>>* ready_queue = nullptr;
  ThreadSafeQueue<CompletedGame>* completed_game_queue = nullptr;
};

class GameRunner {
 public:
  explicit GameRunner(GameRunnerChannels channels);
  ~GameRunner();

  GameRunner(const GameRunner&) = delete;
  GameRunner& operator=(const GameRunner&) = delete;

  void Start();
  void Stop();

 private:
  void RunLoop();

  GameRunnerChannels channels_;
  bool started_ = false;
  bool stopped_ = false;
  mutable std::mutex state_mu_;
  std::thread runner_;
};

}  // namespace engine
