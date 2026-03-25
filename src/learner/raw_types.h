#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "chess/position.h"

namespace engine::learner {

struct RawPolicyEntry {
  std::string move_uci;
  float prob = 0.0f;
};

struct RawPly {
  std::string selected_move_uci;
  std::vector<RawPolicyEntry> policy;
};

struct RawGame {
  std::optional<std::string> initial_fen;
  lczero::GameResult game_result = lczero::GameResult::UNDECIDED;
  std::vector<RawPly> plies;
};

struct RawChunkFile {
  uint32_t version = 0;
  std::vector<RawGame> games;
};

}  // namespace engine::learner
