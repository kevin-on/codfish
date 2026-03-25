#include "learner/storage_facade.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "actor/mcts/output/raw_chunk_format.h"

namespace engine::learner {
namespace {

struct ScopedTempDir {
  ScopedTempDir() {
    path = std::filesystem::temp_directory_path() /
           ("codfish_storage_facade_test_" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
  }

  ~ScopedTempDir() { std::filesystem::remove_all(path); }

  std::filesystem::path path;
};

std::vector<uint8_t> SerializeChunk(
    const std::vector<raw_chunk_format::StoredRawGame>& records) {
  const std::vector<char> header = raw_chunk_format::SerializeChunkHeader();
  std::vector<uint8_t> bytes(header.begin(), header.end());
  for (const raw_chunk_format::StoredRawGame& record : records) {
    const std::vector<char> encoded =
        raw_chunk_format::SerializeChunkRecord(record);
    bytes.insert(bytes.end(), encoded.begin(), encoded.end());
  }
  return bytes;
}

void WriteBytes(const std::filesystem::path& path,
                const std::vector<uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!stream.is_open()) {
    throw std::runtime_error("failed to open storage facade fixture");
  }
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!stream.good()) {
    throw std::runtime_error("failed to write storage facade fixture");
  }
}

raw_chunk_format::StoredRawGame MakeStoredRawGame(
    std::optional<std::string> initial_fen, lczero::GameResult result,
    std::vector<raw_chunk_format::StoredPly> plies) {
  return raw_chunk_format::StoredRawGame{
      .initial_fen = std::move(initial_fen),
      .game_result = result,
      .plies = std::move(plies),
  };
}

TEST(StorageFacade, ReadsChunkFileAndPreservesAllFields) {
  ScopedTempDir temp_dir;
  const std::filesystem::path chunk_path = temp_dir.path / "games.bin";

  const raw_chunk_format::StoredRawGame first = MakeStoredRawGame(
      "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", lczero::GameResult::WHITE_WON,
      {
          raw_chunk_format::StoredPly{
              .selected_move_uci = "e2e4",
              .policy =
                  {
                      raw_chunk_format::StoredPolicyEntry{
                          .move_uci = "e2e4",
                          .prob = 0.75f,
                      },
                      raw_chunk_format::StoredPolicyEntry{
                          .move_uci = "d2d4",
                          .prob = 0.25f,
                      },
                  },
          },
      });
  const raw_chunk_format::StoredRawGame second =
      MakeStoredRawGame(std::nullopt, lczero::GameResult::DRAW, {});

  WriteBytes(chunk_path, SerializeChunk({first, second}));

  const RawChunkFile chunk = ReadRawChunkFile(chunk_path);

  EXPECT_EQ(chunk.version, raw_chunk_format::kChunkVersion);
  ASSERT_EQ(chunk.games.size(), 2u);

  ASSERT_TRUE(chunk.games[0].initial_fen.has_value());
  EXPECT_EQ(*chunk.games[0].initial_fen, *first.initial_fen);
  EXPECT_EQ(chunk.games[0].game_result, first.game_result);
  ASSERT_EQ(chunk.games[0].plies.size(), 1u);
  EXPECT_EQ(chunk.games[0].plies[0].selected_move_uci, "e2e4");
  ASSERT_EQ(chunk.games[0].plies[0].policy.size(), 2u);
  EXPECT_EQ(chunk.games[0].plies[0].policy[0].move_uci, "e2e4");
  EXPECT_FLOAT_EQ(chunk.games[0].plies[0].policy[0].prob, 0.75f);
  EXPECT_EQ(chunk.games[0].plies[0].policy[1].move_uci, "d2d4");
  EXPECT_FLOAT_EQ(chunk.games[0].plies[0].policy[1].prob, 0.25f);

  EXPECT_FALSE(chunk.games[1].initial_fen.has_value());
  EXPECT_EQ(chunk.games[1].game_result, second.game_result);
  EXPECT_TRUE(chunk.games[1].plies.empty());
}

TEST(StorageFacade, PreservesMultiplePliesInOrder) {
  ScopedTempDir temp_dir;
  const std::filesystem::path chunk_path = temp_dir.path / "games.bin";

  const raw_chunk_format::StoredRawGame record = MakeStoredRawGame(
      "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", lczero::GameResult::BLACK_WON,
      {
          raw_chunk_format::StoredPly{
              .selected_move_uci = "e2e4",
              .policy =
                  {
                      raw_chunk_format::StoredPolicyEntry{
                          .move_uci = "e2e4",
                          .prob = 0.6f,
                      },
                  },
          },
          raw_chunk_format::StoredPly{
              .selected_move_uci = "e7e5",
              .policy =
                  {
                      raw_chunk_format::StoredPolicyEntry{
                          .move_uci = "e7e5",
                          .prob = 1.0f,
                      },
                  },
          },
      });

  WriteBytes(chunk_path, SerializeChunk({record}));

  const RawChunkFile chunk = ReadRawChunkFile(chunk_path);

  ASSERT_EQ(chunk.games.size(), 1u);
  ASSERT_EQ(chunk.games[0].plies.size(), 2u);
  EXPECT_EQ(chunk.games[0].plies[0].selected_move_uci, "e2e4");
  EXPECT_EQ(chunk.games[0].plies[1].selected_move_uci, "e7e5");
}

TEST(StorageFacade, ThrowsWhenChunkFileCannotBeOpened) {
  EXPECT_THROW(
      static_cast<void>(ReadRawChunkFile("/definitely/missing/chunk.bin")),
      std::runtime_error);
}

TEST(StorageFacade, ThrowsWhenChunkBytesAreMalformed) {
  ScopedTempDir temp_dir;
  const std::filesystem::path chunk_path = temp_dir.path / "games.bin";
  WriteBytes(chunk_path, std::vector<uint8_t>{'B', 'A', 'D'});

  EXPECT_THROW(static_cast<void>(ReadRawChunkFile(chunk_path)),
               std::runtime_error);
}

}  // namespace
}  // namespace engine::learner

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
