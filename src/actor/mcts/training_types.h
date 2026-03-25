#pragma once

#include <vector>

#include "chess/position.h"

namespace engine {

struct TrainingSampleDraft {
  lczero::PositionHistory root_history;
  lczero::Move selected_move;
  std::vector<lczero::Move> legal_moves;
  std::vector<float> improved_policy;
};

struct CompletedGame {
  std::vector<TrainingSampleDraft> sample_drafts;
  lczero::GameResult game_result = lczero::GameResult::UNDECIDED;
};

}  // namespace engine
