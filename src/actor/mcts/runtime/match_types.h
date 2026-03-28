#pragma once

#include <string>
#include <vector>

#include "actor/mcts/runtime/task_types.h"

namespace engine {

class MatchTask final : public GameTask {
 public:
  int BackendSlotForRequestItem(const EvalRequestItem& item) const override {
    const lczero::Position& position = item.positions[item.len - 1];
    return position.IsBlackToMove() ? black_backend_slot : white_backend_slot;
  }

  int white_backend_slot = 0;
  int black_backend_slot = 1;
  std::vector<std::string> move_uci_history;
};

struct CompletedMatchGame {
  int white_backend_slot = 0;
  int black_backend_slot = 1;
  std::vector<std::string> move_uci_history;
  lczero::GameResult game_result = lczero::GameResult::UNDECIDED;
};

}  // namespace engine
