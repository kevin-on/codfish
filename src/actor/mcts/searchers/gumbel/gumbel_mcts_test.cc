#include "actor/mcts/searchers/gumbel/gumbel_mcts.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "lc0/move_index.h"

namespace engine {
namespace {

EvalResponse MakeRootEvalResponse() {
  EvalResponse response;
  response.items.resize(1);
  response.items[0].policy_logits.assign(lczero::kPolicySize, 0.0f);
  response.items[0].wdl_probs = {0.2f, 0.6f, 0.2f};
  return response;
}

TEST(GumbelMCTS, NonTerminalSearchResultCarriesRootHistoryAndPolicyMetadata) {
  GumbelMCTS searcher(GumbelMCTSConfig{
      .num_action = 1,
      .num_simulation = 1,
      .c_puct = 1.0f,
      .c_visit = 1.0f,
      .c_scale = 1.0f,
  });

  SearchCoroutine coroutine = searcher.Run();
  std::optional<EvalRequest> request = coroutine.next();
  ASSERT_TRUE(request.has_value());
  ASSERT_EQ(request->items.size(), 1u);
  EXPECT_EQ(request->items[0].len, 1);

  std::optional<EvalRequest> next_request = coroutine.send(MakeRootEvalResponse());
  EXPECT_FALSE(next_request.has_value());
  ASSERT_TRUE(coroutine.done());

  SearchResult result = coroutine.take_result();
  EXPECT_EQ(result.game_result, lczero::GameResult::UNDECIDED);
  EXPECT_EQ(result.root_history.GetLength(), 1);
  EXPECT_EQ(lczero::PositionToFen(result.root_history.Last()),
            lczero::PositionToFen(
                lczero::Position::FromFen(lczero::ChessBoard::kStartposFen)));
  ASSERT_FALSE(result.legal_moves.empty());
  ASSERT_EQ(result.legal_moves.size(), result.improved_policy.size());
  EXPECT_NE(std::find(result.legal_moves.begin(), result.legal_moves.end(),
                      result.selected_move),
            result.legal_moves.end());

  const float policy_sum = std::accumulate(result.improved_policy.begin(),
                                           result.improved_policy.end(), 0.0f);
  EXPECT_NEAR(policy_sum, 1.0f, 1e-5f);
  for (float prob : result.improved_policy) {
    EXPECT_GE(prob, 0.0f);
  }
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
