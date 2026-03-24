#pragma once

#include <array>
#include <vector>

#include "chess/position.h"
#include "chess/types.h"
#include "engine/encoder.h"
#include "request_reply_coroutine.h"

namespace engine {

struct EvalRequestItem {
  std::array<lczero::Position, kHistoryLength>
      positions;  // positions[0] = oldest, positions[len - 1] = current.
  uint8_t len;    // number of valid positions
};

struct EvalResponseItem {
  std::vector<float> policy_logits;  // kPolicySize
  std::vector<float> wdl_probs;
};

struct EvalRequest {
  std::vector<EvalRequestItem> items;
};

struct EvalResponse {
  std::vector<EvalResponseItem> items;
};

struct SearchResult {
  // If game_result != UNDECIDED, the search position is terminal and
  // selected_move, legal_moves, and improved_policy are invalid.
  lczero::Move selected_move;
  std::vector<lczero::Move> legal_moves;
  std::vector<float> improved_policy;
  lczero::GameResult game_result = lczero::GameResult::UNDECIDED;
};

using SearchCoroutine =
    RequestReplyCoroutine<EvalRequest, EvalResponse, SearchResult>;

class MCTSSearcher {
 public:
  virtual ~MCTSSearcher() = default;
  virtual SearchCoroutine Run() = 0;
  virtual void CommitMove(lczero::Move move) = 0;
};

}  // namespace engine
