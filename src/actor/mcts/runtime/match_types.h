#pragma once

#include <memory>
#include <string>
#include <vector>

#include "actor/mcts/runtime/task_types.h"

namespace engine {

class MatchTask final : public GameTask {
 public:
  int BackendSlotForRequestItem(const EvalRequestItem& item) const override {
    static_cast<void>(item);
    return active_player_is_white ? white_backend_slot : black_backend_slot;
  }

  std::unique_ptr<MCTSSearcher> inactive_searcher;
  bool active_player_is_white = true;
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
