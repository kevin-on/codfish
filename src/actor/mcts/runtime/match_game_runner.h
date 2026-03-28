#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "actor/mcts/primitives/thread_safe_queue.h"
#include "actor/mcts/runtime/match_types.h"

namespace engine {

struct MatchGameRunnerChannels {
  ThreadSafeQueue<CompletedSearch>* completion_queue = nullptr;
  ThreadSafeQueue<std::unique_ptr<GameTask>>* ready_queue = nullptr;
  std::function<void(CompletedMatchGame)> on_completed_game;
};

class MatchGameRunner {
 public:
  explicit MatchGameRunner(MatchGameRunnerChannels channels);
  ~MatchGameRunner();

  MatchGameRunner(const MatchGameRunner&) = delete;
  MatchGameRunner& operator=(const MatchGameRunner&) = delete;

  void Start();
  void Stop();

 private:
  void RunLoop();

  MatchGameRunnerChannels channels_;
  bool started_ = false;
  bool stopped_ = false;
  mutable std::mutex state_mu_;
  std::thread runner_;
};

}  // namespace engine
