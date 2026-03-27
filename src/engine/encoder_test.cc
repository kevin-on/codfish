#include "encoder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

namespace engine {
namespace {

std::span<const uint8_t> PositionSlot(std::span<const uint8_t> input,
                                      int history_idx) {
  const std::size_t offset =
      static_cast<std::size_t>(history_idx) * kPositionPlanes * kBoardArea;
  return input.subspan(offset, kPositionPlanes * kBoardArea);
}

std::span<const uint8_t> GlobalSlot(std::span<const uint8_t> input) {
  const std::size_t offset =
      static_cast<std::size_t>(kHistoryLength) * kPositionPlanes * kBoardArea;
  return input.subspan(offset, kGlobalPlanes * kBoardArea);
}

std::span<const uint8_t> Plane(std::span<const uint8_t> block, int plane_idx) {
  const std::size_t offset = static_cast<std::size_t>(plane_idx) * kBoardArea;
  return block.subspan(offset, kBoardArea);
}

bool AllValues(std::span<const uint8_t> values, uint8_t expected) {
  return std::all_of(values.begin(), values.end(),
                     [expected](uint8_t value) { return value == expected; });
}

int Sum(std::span<const uint8_t> values) {
  return std::accumulate(values.begin(), values.end(), 0);
}

std::array<uint8_t, kInputElements> Encode(
    const lczero::PositionHistory& history) {
  FeatureEncoder encoder;
  std::array<uint8_t, kInputElements> out{};
  encoder.EncodeOne(history, out);
  return out;
}

std::array<uint8_t, kInputElements> Encode(
    std::span<const lczero::Position> positions) {
  FeatureEncoder encoder;
  std::array<uint8_t, kInputElements> out{};
  encoder.EncodeOne(positions, out);
  return out;
}

std::vector<uint8_t> EncodeBatch(
    std::span<const std::span<const lczero::Position>> batch) {
  FeatureEncoder encoder;
  std::vector<uint8_t> out(kInputElements * batch.size());
  encoder.EncodeBatch(batch, out);
  return out;
}

TEST(FeatureEncoder, EncodeOneUsesLatestPositionFirstAndPadsHistory) {
  const std::array<lczero::Position, 2> positions = {
      lczero::Position::FromFen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"),
      lczero::Position::FromFen("4k3/8/8/8/8/8/8/4K2R w - - 0 1"),
  };
  const lczero::PositionHistory history(positions);
  const auto out = Encode(history);

  const auto latest_rooks = Plane(PositionSlot(out, 0), kOurRooksPlane);
  EXPECT_EQ(latest_rooks[lczero::Square::Parse("h1").as_idx()], 1);
  EXPECT_EQ(Sum(latest_rooks), 1);

  const auto previous_rooks = Plane(PositionSlot(out, 1), kOurRooksPlane);
  EXPECT_EQ(previous_rooks[lczero::Square::Parse("a1").as_idx()], 1);
  EXPECT_EQ(Sum(previous_rooks), 1);

  EXPECT_TRUE(AllValues(PositionSlot(out, 2), 0));
}

TEST(FeatureEncoder, EncodeOneEncodesGlobalAndRepetitionPlanes) {
  auto position = lczero::Position::FromFen(
      "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 17 2");
  position.SetRepetitions(2, 4);
  lczero::PositionHistory history;
  history.Reset(position);

  const auto out = Encode(history);
  const auto global = GlobalSlot(out);

  EXPECT_TRUE(AllValues(Plane(global, kOurKingsideCastlingPlane), 1));
  EXPECT_TRUE(AllValues(Plane(global, kOurQueensideCastlingPlane), 1));
  EXPECT_TRUE(AllValues(Plane(global, kTheirKingsideCastlingPlane), 1));
  EXPECT_TRUE(AllValues(Plane(global, kTheirQueensideCastlingPlane), 1));

  const auto en_passant = Plane(global, kEnPassantPlane);
  EXPECT_EQ(en_passant[lczero::Square::Parse("c6").as_idx()], 1);
  EXPECT_EQ(Sum(en_passant), 1);

  EXPECT_TRUE(AllValues(Plane(global, kRule50Plane), 17));
  EXPECT_TRUE(
      AllValues(Plane(PositionSlot(out, 0), kRepetitionAtLeast1Plane), 1));
  EXPECT_TRUE(
      AllValues(Plane(PositionSlot(out, 0), kRepetitionAtLeast2Plane), 1));
}

TEST(FeatureEncoder, EncodeBatchMatchesEncodeOneOutputs) {
  const std::array<lczero::Position, 2> history_positions = {
      lczero::Position::FromFen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"),
      lczero::Position::FromFen("4k3/8/8/8/8/8/8/4K2R w - - 0 1"),
  };
  lczero::Position repeated =
      lczero::Position::FromFen("8/8/8/8/8/8/4k3/4K3 w - - 3 1");
  repeated.SetRepetitions(1, 2);

  const std::array<lczero::PositionHistory, 2> batch = {
      lczero::PositionHistory(history_positions),
      lczero::PositionHistory(std::span<const lczero::Position>(&repeated, 1)),
  };

  FeatureEncoder encoder;
  std::array<uint8_t, 2 * kInputElements> actual{};
  std::array<uint8_t, kInputElements> expected0{};
  std::array<uint8_t, kInputElements> expected1{};

  encoder.EncodeBatch(batch, actual);
  encoder.EncodeOne(batch[0], expected0);
  encoder.EncodeOne(batch[1], expected1);

  EXPECT_TRUE(std::equal(expected0.begin(), expected0.end(), actual.begin()));
  EXPECT_TRUE(std::equal(expected1.begin(), expected1.end(),
                         actual.begin() + kInputElements));
}

TEST(FeatureEncoder, EncodeOneSpanMatchesPositionHistoryOverload) {
  const std::array<lczero::Position, 3> positions = {
      lczero::Position::FromFen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"),
      lczero::Position::FromFen("4k3/8/8/8/8/8/8/4K2R w - - 0 1"),
      lczero::Position::FromFen("4k3/8/8/8/8/8/3K4/7R w - - 0 1"),
  };
  const lczero::PositionHistory history(positions);

  const auto from_history = Encode(history);
  const auto from_span = Encode(std::span<const lczero::Position>(positions));

  EXPECT_TRUE(
      std::equal(from_history.begin(), from_history.end(), from_span.begin()));
}

TEST(FeatureEncoder, EncodeBatchSpanMatchesEncodeOneOutputs) {
  const std::array<lczero::Position, 2> first = {
      lczero::Position::FromFen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"),
      lczero::Position::FromFen("4k3/8/8/8/8/8/8/4K2R w - - 0 1"),
  };
  const std::array<lczero::Position, 1> second = {
      lczero::Position::FromFen("8/8/8/8/8/8/4k3/4K3 w - - 3 1"),
  };
  const std::array<std::span<const lczero::Position>, 2> batch = {
      std::span<const lczero::Position>(first),
      std::span<const lczero::Position>(second),
  };

  const std::vector<uint8_t> actual = EncodeBatch(batch);
  const auto expected0 = Encode(batch[0]);
  const auto expected1 = Encode(batch[1]);

  EXPECT_TRUE(std::equal(expected0.begin(), expected0.end(), actual.begin()));
  EXPECT_TRUE(std::equal(expected1.begin(), expected1.end(),
                         actual.begin() + kInputElements));
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
