#pragma once

#include <memory>

#include "actor/mcts/searcher.h"

namespace engine {

struct GumbelMCTSConfig {
  int num_action = 0;      // Number of initial candidates in Gumbel algorithm
  int num_simulation = 0;  // Total number of simulations at the root
  float c_puct = 0.0f;     // Exploration constant for non-root AlphaZero-style
                           // PUCT.
  float c_visit = 0.0f;    // c_visit in Eq (8) in the gumbel alphazero paper
  float c_scale = 0.0f;    // c_scale in Eq (8) in the gumbel alphazero paper
};

class GumbelMCTS : public MCTSSearcher {
 public:
  explicit GumbelMCTS(GumbelMCTSConfig config);
  ~GumbelMCTS() override;
  SearchCoroutine Run() override;
  void CommitMove(lczero::Move move) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace engine
