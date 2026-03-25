#include "learner/sample_facade.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "chess/position.h"
#include "engine/encoder.h"

namespace engine::learner {
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

template <typename T, std::size_t N>
void ExpectSliceEq(const std::vector<T>& actual, std::size_t offset,
                   const std::array<T, N>& expected) {
  ASSERT_GE(actual.size(), offset + expected.size());
  EXPECT_TRUE(
      std::equal(expected.begin(), expected.end(), actual.begin() + offset));
}

TEST(SampleFacadePublic, ExposesMetadataAndBufferSizes) {
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

  const EncodedGameSamples samples = EncodeRawGame(raw_game);

  ASSERT_EQ(samples.sample_count, 1);
  EXPECT_EQ(samples.input_channels, kInputPlanes);
  EXPECT_EQ(samples.policy_size, lczero::kPolicySize);
  EXPECT_EQ(samples.inputs.size(), kInputElements);
  EXPECT_EQ(samples.policy_targets.size(),
            static_cast<std::size_t>(lczero::kPolicySize));
  EXPECT_EQ(samples.wdl_targets.size(), 3u);
  EXPECT_EQ(samples.wdl_targets,
            (std::vector<float>{1.0f, 0.0f, 0.0f}));

  const lczero::PositionHistory history = MakeHistory(*raw_game.initial_fen);
  const auto expected_input = EncodeHistory(history);
  ExpectSliceEq(samples.inputs, 0, expected_input);

  const lczero::Move move = history.Last().GetBoard().ParseMove("e2e4");
  const lczero::Move alt = history.Last().GetBoard().ParseMove("e2e3");
  EXPECT_FLOAT_EQ(samples.policy_targets[lczero::MoveToNNIndex(move, 0)], 0.75f);
  EXPECT_FLOAT_EQ(samples.policy_targets[lczero::MoveToNNIndex(alt, 0)], 0.25f);
}

TEST(SampleFacadePublic, PreservesFlatteningOrderAcrossPlies) {
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

  const EncodedGameSamples samples = EncodeRawGame(raw_game);

  ASSERT_EQ(samples.sample_count, 2);
  EXPECT_EQ(samples.inputs.size(), 2u * kInputElements);
  EXPECT_EQ(samples.policy_targets.size(),
            2u * static_cast<std::size_t>(lczero::kPolicySize));
  EXPECT_EQ(samples.wdl_targets.size(), 6u);

  const auto first_history = MakeHistory(*raw_game.initial_fen);
  const auto second_history = MakeHistory(*raw_game.initial_fen, {"e2e4"});
  ExpectSliceEq(samples.inputs, 0, EncodeHistory(first_history));
  ExpectSliceEq(samples.inputs, kInputElements, EncodeHistory(second_history));
  EXPECT_EQ(samples.wdl_targets,
            (std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f}));

  const lczero::Move first_move = first_history.Last().GetBoard().ParseMove("e2e4");
  const lczero::Move second_move =
      second_history.Last().GetBoard().ParseMove("d7d5");
  EXPECT_FLOAT_EQ(
      samples.policy_targets[lczero::MoveToNNIndex(first_move, 0)], 1.0f);
  EXPECT_FLOAT_EQ(
      samples.policy_targets[lczero::kPolicySize +
                             lczero::MoveToNNIndex(second_move, 0)],
      1.0f);
}

TEST(SampleFacadePublic, ReturnsMetadataOnlyForValidEmptyGame) {
  const EncodedGameSamples samples =
      EncodeRawGame(MakeRawGame(std::nullopt, lczero::GameResult::DRAW, {}));

  EXPECT_EQ(samples.sample_count, 0);
  EXPECT_EQ(samples.input_channels, kInputPlanes);
  EXPECT_EQ(samples.policy_size, lczero::kPolicySize);
  EXPECT_TRUE(samples.inputs.empty());
  EXPECT_TRUE(samples.policy_targets.empty());
  EXPECT_TRUE(samples.wdl_targets.empty());
}

TEST(SampleFacadePublic, RejectsInvalidEmptyGame) {
  EXPECT_THROW(static_cast<void>(
                   EncodeRawGame(MakeRawGame(
                       std::nullopt, lczero::GameResult::UNDECIDED, {}))),
               std::runtime_error);
}

TEST(SampleFacadePublic, SurfacesInvalidRawGameErrors) {
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

  EXPECT_THROW(static_cast<void>(EncodeRawGame(raw_game)), std::runtime_error);
}

}  // namespace
}  // namespace engine::learner

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
