#pragma once

#include "actor/mcts/search_types.h"

namespace engine {

class MCTSSearcher {
 public:
  virtual ~MCTSSearcher() = default;
  virtual SearchCoroutine Run() = 0;
  virtual void CommitMove(lczero::Move move) = 0;
};

}  // namespace engine
