#include "learner/sample_facade_internal.h"

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "chess/position.h"
#include "engine/encoder.h"

namespace engine::learner::internal {
namespace {

RawGame MakeRawGame(std::optional<std::string> initial_fen,
                    lczero::GameResult game_result,
                    std::vector<RawPly> plies) {
  return RawGame{
      .initial_fen = std::move(initial_fen),
      .game_result = game_result,
      .plies = std::move(plies),
  };
}

lczero::PositionHistory MakeHistory(
    std::string_view initial_fen,
    std::initializer_list<std::string_view> moves = {}) {
  lczero::PositionHistory history;
  history.Reset(lczero::Position::FromFen(initial_fen));
  for (const std::string_view move_uci : moves) {
    const lczero::Position current = history.Last();
    history.Append(current.GetBoard().ParseMove(move_uci));
  }
  return history;
}

std::array<uint8_t, kInputElements> EncodeHistory(
    const lczero::PositionHistory& history) {
  FeatureEncoder encoder;
  std::array<uint8_t, kInputElements> out{};
  encoder.EncodeOne(history, out);
  return out;
}

TEST(SampleFacadeInternal, BuildsWhiteToMoveSinglePlySample) {
  const RawGame raw_game = MakeRawGame(
      "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", lczero::GameResult::WHITE_WON,
      {
          RawPly{
              .selected_move_uci = "e2e4",
              .policy =
                  {
                      RawPolicyEntry{.move_uci = "e2e4", .prob = 0.75f},
                      RawPolicyEntry{.move_uci = "e2e3", .prob = 0.25f},
                  },
          },
      });

  const std::vector<EncodedSampleDraft> samples =
      BuildEncodedSampleDrafts(raw_game);

  ASSERT_EQ(samples.size(), 1u);
  EXPECT_EQ(samples[0].selected_move_uci, "e2e4");
  EXPECT_EQ(samples[0].wdl_target,
            (std::array<float, 3>{1.0f, 0.0f, 0.0f}));

  const lczero::PositionHistory history = MakeHistory(*raw_game.initial_fen);
  const auto expected_input = EncodeHistory(history);
  EXPECT_EQ(samples[0].input, expected_input);

  const lczero::Move move = history.Last().GetBoard().ParseMove("e2e4");
  const lczero::Move alt = history.Last().GetBoard().ParseMove("e2e3");
  EXPECT_FLOAT_EQ(samples[0].policy_target[lczero::MoveToNNIndex(move, 0)],
                  0.75f);
  EXPECT_FLOAT_EQ(samples[0].policy_target[lczero::MoveToNNIndex(alt, 0)],
                  0.25f);
}

TEST(SampleFacadeInternal, BuildsBlackToMoveAbsoluteUciSample) {
  const RawGame raw_game = MakeRawGame(
      "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
      lczero::GameResult::BLACK_WON,
      {
          RawPly{
              .selected_move_uci = "e7e5",
              .policy =
                  {
                      RawPolicyEntry{.move_uci = "e7e5", .prob = 1.0f},
                  },
          },
      });

  const std::vector<EncodedSampleDraft> samples =
      BuildEncodedSampleDrafts(raw_game);

  ASSERT_EQ(samples.size(), 1u);
  EXPECT_EQ(samples[0].selected_move_uci, "e7e5");
  EXPECT_EQ(samples[0].wdl_target,
            (std::array<float, 3>{1.0f, 0.0f, 0.0f}));

  const lczero::PositionHistory history = MakeHistory(*raw_game.initial_fen);
  const lczero::Move move = history.Last().GetBoard().ParseMove("e7e5");
  EXPECT_FLOAT_EQ(samples[0].policy_target[lczero::MoveToNNIndex(move, 0)],
                  1.0f);
}

TEST(SampleFacadeInternal, PreservesHistoryProgressionAcrossPlies) {
  const RawGame raw_game = MakeRawGame(
      "4k3/3p4/8/8/8/8/4P3/4K3 w - - 0 1", lczero::GameResult::DRAW,
      {
          RawPly{
              .selected_move_uci = "e2e4",
              .policy =
                  {
                      RawPolicyEntry{.move_uci = "e2e4", .prob = 1.0f},
                  },
          },
          RawPly{
              .selected_move_uci = "d7d5",
              .policy =
                  {
                      RawPolicyEntry{.move_uci = "d7d5", .prob = 1.0f},
                  },
          },
      });

  const std::vector<EncodedSampleDraft> samples =
      BuildEncodedSampleDrafts(raw_game);

  ASSERT_EQ(samples.size(), 2u);
  EXPECT_EQ(samples[0].selected_move_uci, "e2e4");
  EXPECT_EQ(samples[1].selected_move_uci, "d7d5");
  EXPECT_EQ(samples[0].wdl_target,
            (std::array<float, 3>{0.0f, 1.0f, 0.0f}));
  EXPECT_EQ(samples[1].wdl_target,
            (std::array<float, 3>{0.0f, 1.0f, 0.0f}));

  const lczero::PositionHistory first_history = MakeHistory(*raw_game.initial_fen);
  const lczero::PositionHistory second_history =
      MakeHistory(*raw_game.initial_fen, {"e2e4"});
  EXPECT_EQ(samples[0].input, EncodeHistory(first_history));
  EXPECT_EQ(samples[1].input, EncodeHistory(second_history));
}

TEST(SampleFacadeInternal, ReturnsEmptyWhenGameHasNoPlies) {
  const RawGame raw_game =
      MakeRawGame(std::nullopt, lczero::GameResult::DRAW, {});

  EXPECT_TRUE(BuildEncodedSampleDrafts(raw_game).empty());
}

TEST(SampleFacadeInternal, ThrowsWhenInitialFenIsMissing) {
  const RawGame raw_game = MakeRawGame(
      std::nullopt, lczero::GameResult::DRAW,
      {
          RawPly{
              .selected_move_uci = "e2e4",
              .policy =
                  {
                      RawPolicyEntry{.move_uci = "e2e4", .prob = 1.0f},
                  },
          },
      });

  EXPECT_THROW(static_cast<void>(BuildEncodedSampleDrafts(raw_game)),
               std::runtime_error);
}

TEST(SampleFacadeInternal, ThrowsWhenGameResultIsUndecided) {
  const RawGame raw_game = MakeRawGame(
      "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
      lczero::GameResult::UNDECIDED,
      {
          RawPly{
              .selected_move_uci = "e2e4",
              .policy =
                  {
                      RawPolicyEntry{.move_uci = "e2e4", .prob = 1.0f},
                  },
          },
      });

  EXPECT_THROW(static_cast<void>(BuildEncodedSampleDrafts(raw_game)),
               std::runtime_error);
}

TEST(SampleFacadeInternal, ThrowsWhenSelectedMoveIsInvalid) {
  const RawGame raw_game = MakeRawGame(
      "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", lczero::GameResult::DRAW,
      {
          RawPly{
              .selected_move_uci = "zzzz",
              .policy =
                  {
                      RawPolicyEntry{.move_uci = "e2e4", .prob = 1.0f},
                  },
          },
      });

  EXPECT_THROW(static_cast<void>(BuildEncodedSampleDrafts(raw_game)),
               std::runtime_error);
}

TEST(SampleFacadeInternal, ThrowsWhenPolicyMoveIsIllegal) {
  const RawGame raw_game = MakeRawGame(
      "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", lczero::GameResult::DRAW,
      {
          RawPly{
              .selected_move_uci = "e2e4",
              .policy =
                  {
                      RawPolicyEntry{.move_uci = "e2e4", .prob = 0.5f},
                      RawPolicyEntry{.move_uci = "e2e5", .prob = 0.5f},
                  },
          },
      });

  EXPECT_THROW(static_cast<void>(BuildEncodedSampleDrafts(raw_game)),
               std::runtime_error);
}

TEST(SampleFacadeInternal, ThrowsWhenPolicyEntriesDuplicateAnIndex) {
  const RawGame raw_game = MakeRawGame(
      "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", lczero::GameResult::DRAW,
      {
          RawPly{
              .selected_move_uci = "e2e4",
              .policy =
                  {
                      RawPolicyEntry{.move_uci = "e2e4", .prob = 0.5f},
                      RawPolicyEntry{.move_uci = "e2e4", .prob = 0.5f},
                  },
          },
      });

  EXPECT_THROW(static_cast<void>(BuildEncodedSampleDrafts(raw_game)),
               std::runtime_error);
}

TEST(SampleFacadeInternal, ThrowsWhenSelectedMoveIsAbsentFromPolicy) {
  const RawGame raw_game = MakeRawGame(
      "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", lczero::GameResult::DRAW,
      {
          RawPly{
              .selected_move_uci = "e2e4",
              .policy =
                  {
                      RawPolicyEntry{.move_uci = "e2e3", .prob = 1.0f},
                  },
          },
      });

  EXPECT_THROW(static_cast<void>(BuildEncodedSampleDrafts(raw_game)),
               std::runtime_error);
}

}  // namespace
}  // namespace engine::learner::internal

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
