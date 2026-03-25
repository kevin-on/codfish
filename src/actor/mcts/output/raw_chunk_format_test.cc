#include "actor/mcts/output/raw_chunk_format.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine {
namespace {

using raw_chunk_format::ParsedChunk;
using raw_chunk_format::StoredPly;
using raw_chunk_format::StoredPolicyEntry;
using raw_chunk_format::StoredRawGame;

std::vector<uint8_t> SerializeChunk(const std::vector<StoredRawGame>& records) {
  const std::vector<char> header = raw_chunk_format::SerializeChunkHeader();
  std::vector<uint8_t> bytes(header.begin(), header.end());
  for (const StoredRawGame& record : records) {
    const std::vector<char> encoded =
        raw_chunk_format::SerializeChunkRecord(record);
    bytes.insert(bytes.end(), encoded.begin(), encoded.end());
  }
  return bytes;
}

void ExpectStoredRawGameEq(const StoredRawGame& actual,
                           const StoredRawGame& expected) {
  EXPECT_EQ(actual.initial_fen, expected.initial_fen);
  EXPECT_EQ(actual.game_result, expected.game_result);
  ASSERT_EQ(actual.plies.size(), expected.plies.size());
  for (std::size_t i = 0; i < expected.plies.size(); ++i) {
    const StoredPly& actual_ply = actual.plies[i];
    const StoredPly& expected_ply = expected.plies[i];
    EXPECT_EQ(actual_ply.selected_move_raw, expected_ply.selected_move_raw);
    ASSERT_EQ(actual_ply.policy.size(), expected_ply.policy.size());
    for (std::size_t j = 0; j < expected_ply.policy.size(); ++j) {
      const StoredPolicyEntry& actual_entry = actual_ply.policy[j];
      const StoredPolicyEntry& expected_entry = expected_ply.policy[j];
      EXPECT_EQ(actual_entry.move_raw, expected_entry.move_raw);
      EXPECT_FLOAT_EQ(actual_entry.prob, expected_entry.prob);
    }
  }
}

StoredRawGame MakeStoredRawGame(std::optional<std::string> initial_fen,
                                lczero::GameResult result,
                                std::vector<StoredPly> plies) {
  return StoredRawGame{
      .initial_fen = std::move(initial_fen),
      .game_result = result,
      .plies = std::move(plies),
  };
}

TEST(RawChunkFormat, RoundTripsSingleGameRecord) {
  const StoredRawGame expected = MakeStoredRawGame(
      "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", lczero::GameResult::WHITE_WON,
      {
          StoredPly{
              .selected_move_raw = 101,
              .policy =
                  {
                      StoredPolicyEntry{.move_raw = 101, .prob = 0.75f},
                      StoredPolicyEntry{.move_raw = 102, .prob = 0.25f},
                  },
          },
          StoredPly{
              .selected_move_raw = 203,
              .policy =
                  {
                      StoredPolicyEntry{.move_raw = 203, .prob = 1.0f},
                  },
          },
      });

  const std::vector<uint8_t> bytes = SerializeChunk({expected});
  const ParsedChunk parsed = raw_chunk_format::ParseChunk(bytes);

  EXPECT_EQ(parsed.version, raw_chunk_format::kChunkVersion);
  ASSERT_EQ(parsed.records.size(), 1u);
  ExpectStoredRawGameEq(parsed.records[0], expected);
}

TEST(RawChunkFormat, RoundTripsChunkWithMultipleRecords) {
  const StoredRawGame first = MakeStoredRawGame(
      "startpos fen", lczero::GameResult::DRAW,
      {
          StoredPly{
              .selected_move_raw = 17,
              .policy =
                  {
                      StoredPolicyEntry{.move_raw = 17, .prob = 0.6f},
                      StoredPolicyEntry{.move_raw = 18, .prob = 0.4f},
                  },
          },
      });
  const StoredRawGame second =
      MakeStoredRawGame(std::nullopt, lczero::GameResult::BLACK_WON, {});

  const std::vector<uint8_t> bytes = SerializeChunk({first, second});
  const ParsedChunk parsed = raw_chunk_format::ParseChunk(bytes);

  EXPECT_EQ(parsed.version, raw_chunk_format::kChunkVersion);
  ASSERT_EQ(parsed.records.size(), 2u);
  ExpectStoredRawGameEq(parsed.records[0], first);
  ExpectStoredRawGameEq(parsed.records[1], second);
}

TEST(RawChunkFormat, RejectsBadMagic) {
  const StoredRawGame record =
      MakeStoredRawGame(std::nullopt, lczero::GameResult::DRAW, {});
  std::vector<uint8_t> bytes = SerializeChunk({record});
  bytes[0] = static_cast<uint8_t>('X');

  EXPECT_THROW(static_cast<void>(raw_chunk_format::ParseChunk(bytes)),
               std::runtime_error);
}

TEST(RawChunkFormat, RejectsTruncatedRecord) {
  const StoredRawGame record = MakeStoredRawGame(
      "fen", lczero::GameResult::WHITE_WON,
      {
          StoredPly{
              .selected_move_raw = 9,
              .policy =
                  {
                      StoredPolicyEntry{.move_raw = 9, .prob = 1.0f},
                  },
          },
      });
  std::vector<uint8_t> bytes = SerializeChunk({record});
  bytes.pop_back();

  EXPECT_THROW(static_cast<void>(raw_chunk_format::ParseChunk(bytes)),
               std::runtime_error);
}

TEST(RawChunkFormat, RejectsCorruptedRecordLength) {
  const StoredRawGame record = MakeStoredRawGame(
      "fen", lczero::GameResult::DRAW,
      {
          StoredPly{
              .selected_move_raw = 31,
              .policy =
                  {
                      StoredPolicyEntry{.move_raw = 31, .prob = 1.0f},
                  },
          },
      });
  std::vector<uint8_t> bytes = SerializeChunk({record});
  const std::size_t record_len_offset = raw_chunk_format::kChunkMagic.size() +
                                        sizeof(raw_chunk_format::kChunkVersion);
  bytes[record_len_offset] = 0x01;
  bytes[record_len_offset + 1] = 0x00;
  bytes[record_len_offset + 2] = 0x00;
  bytes[record_len_offset + 3] = 0x00;

  EXPECT_THROW(static_cast<void>(raw_chunk_format::ParseChunk(bytes)),
               std::runtime_error);
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
