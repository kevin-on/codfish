#include "chunk_writer_runtime.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "chess/position.h"
#include "raw_chunk_format.h"

namespace engine {
namespace {

struct ScopedTempDir {
  ScopedTempDir() {
    path = std::filesystem::temp_directory_path() /
           ("codfish_chunk_writer_test_" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
  }

  ~ScopedTempDir() { std::filesystem::remove_all(path); }

  std::filesystem::path path;
};

std::vector<uint8_t> ReadFileBytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream.is_open()) {
    throw std::runtime_error("failed to open chunk file");
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream),
                              std::istreambuf_iterator<char>());
}

raw_chunk_format::ParsedChunk ReadChunk(const std::filesystem::path& path) {
  const std::vector<uint8_t> bytes = ReadFileBytes(path);
  return raw_chunk_format::ParseChunk(bytes);
}

uint16_t FirstSelectedMoveRaw(const raw_chunk_format::ParsedChunk& chunk) {
  if (chunk.records.empty() || chunk.records.front().plies.empty()) {
    throw std::runtime_error("missing first selected move");
  }
  return chunk.records.front().plies.front().selected_move_raw;
}

lczero::Move ParseMove(const lczero::Position& position, const char* uci) {
  return position.GetBoard().ParseMove(uci);
}

lczero::PositionHistory MakeHistory(
    std::string_view fen, std::initializer_list<const char*> moves = {}) {
  lczero::PositionHistory history;
  history.Reset(lczero::Position::FromFen(fen));
  for (const char* uci : moves) {
    const lczero::Position current = history.Last();
    history.Append(ParseMove(current, uci));
  }
  return history;
}

CompletedGame MakeSinglePlyCompletedGame(std::string_view fen,
                                         const char* selected_move_uci,
                                         lczero::GameResult game_result) {
  const lczero::PositionHistory history = MakeHistory(fen);
  const lczero::Move selected_move =
      ParseMove(history.Last(), selected_move_uci);
  return CompletedGame{
      .sample_drafts =
          {
              TrainingSampleDraft{
                  .root_history = history,
                  .selected_move = selected_move,
                  .legal_moves = {selected_move},
                  .improved_policy = {1.0f},
              },
          },
      .game_result = game_result,
  };
}

TEST(ChunkWriterRuntime, WritesCompletedGamesIntoChunkFile) {
  ScopedTempDir temp_dir;
  ThreadSafeQueue<CompletedGame> completed_game_queue;
  ChunkWriterRuntime writer(
      ChunkWriterChannels{.completed_game_queue = &completed_game_queue},
      ChunkWriterOptions{.output_dir = temp_dir.path});
  writer.Start();

  constexpr std::string_view kFen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1";
  const lczero::Move expected_first_move =
      ParseMove(MakeHistory(kFen).Last(), "e2e3");

  ASSERT_TRUE(completed_game_queue.push(
      MakeSinglePlyCompletedGame(kFen, "e2e3", lczero::GameResult::WHITE_WON)));
  ASSERT_TRUE(completed_game_queue.push(CompletedGame{
      .sample_drafts = {},
      .game_result = lczero::GameResult::DRAW,
  }));

  writer.Stop();

  const std::filesystem::path chunk_path =
      temp_dir.path / raw_chunk_format::ChunkFileName(1);
  ASSERT_TRUE(std::filesystem::exists(chunk_path));

  const raw_chunk_format::ParsedChunk chunk = ReadChunk(chunk_path);
  EXPECT_EQ(chunk.version, raw_chunk_format::kChunkVersion);
  ASSERT_EQ(chunk.records.size(), 2u);
  EXPECT_EQ(FirstSelectedMoveRaw(chunk), expected_first_move.raw_data());
  EXPECT_EQ(chunk.records[0].game_result, lczero::GameResult::WHITE_WON);
  EXPECT_FALSE(chunk.records[1].initial_fen.has_value());
  EXPECT_EQ(chunk.records[1].game_result, lczero::GameResult::DRAW);
  EXPECT_TRUE(chunk.records[1].plies.empty());
}

TEST(ChunkWriterRuntime, RotatesChunkFilesWhenMaxChunkBytesIsExceeded) {
  ScopedTempDir temp_dir;
  ThreadSafeQueue<CompletedGame> completed_game_queue;
  ChunkWriterRuntime writer(
      ChunkWriterChannels{.completed_game_queue = &completed_game_queue},
      ChunkWriterOptions{
          .output_dir = temp_dir.path,
          .max_chunk_bytes = 1,
      });
  writer.Start();

  constexpr std::string_view kFirstFen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1";
  constexpr std::string_view kSecondFen = "4k3/8/8/8/8/8/3P4/4K3 w - - 0 1";
  const lczero::Move first_selected =
      ParseMove(MakeHistory(kFirstFen).Last(), "e2e3");
  const lczero::Move second_selected =
      ParseMove(MakeHistory(kSecondFen).Last(), "d2d3");
  ASSERT_TRUE(completed_game_queue.push(MakeSinglePlyCompletedGame(
      kFirstFen, "e2e3", lczero::GameResult::WHITE_WON)));
  ASSERT_TRUE(completed_game_queue.push(MakeSinglePlyCompletedGame(
      kSecondFen, "d2d3", lczero::GameResult::DRAW)));

  writer.Stop();

  const std::filesystem::path first_chunk =
      temp_dir.path / raw_chunk_format::ChunkFileName(1);
  const std::filesystem::path second_chunk =
      temp_dir.path / raw_chunk_format::ChunkFileName(2);
  ASSERT_TRUE(std::filesystem::exists(first_chunk));
  ASSERT_TRUE(std::filesystem::exists(second_chunk));

  const raw_chunk_format::ParsedChunk first_parsed = ReadChunk(first_chunk);
  const raw_chunk_format::ParsedChunk second_parsed = ReadChunk(second_chunk);
  ASSERT_EQ(first_parsed.records.size(), 1u);
  ASSERT_EQ(second_parsed.records.size(), 1u);
  EXPECT_EQ(FirstSelectedMoveRaw(first_parsed), first_selected.raw_data());
  EXPECT_EQ(FirstSelectedMoveRaw(second_parsed), second_selected.raw_data());
}

TEST(ChunkWriterRuntime, StopIsSafeWithNoGames) {
  ScopedTempDir temp_dir;
  ThreadSafeQueue<CompletedGame> completed_game_queue;
  ChunkWriterRuntime writer(
      ChunkWriterChannels{.completed_game_queue = &completed_game_queue},
      ChunkWriterOptions{.output_dir = temp_dir.path});

  writer.Start();
  writer.Stop();

  EXPECT_FALSE(std::filesystem::exists(temp_dir.path /
                                       raw_chunk_format::ChunkFileName(1)));
}

TEST(ChunkWriterRuntime, PreservesCompletedGameOrderAcrossChunks) {
  ScopedTempDir temp_dir;
  ThreadSafeQueue<CompletedGame> completed_game_queue;
  ChunkWriterRuntime writer(
      ChunkWriterChannels{.completed_game_queue = &completed_game_queue},
      ChunkWriterOptions{
          .output_dir = temp_dir.path,
          .max_chunk_bytes = 1,
      });
  writer.Start();

  constexpr std::string_view kFirstFen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1";
  constexpr std::string_view kSecondFen = "4k3/8/8/8/8/8/3P4/4K3 w - - 0 1";
  constexpr std::string_view kThirdFen = "4k3/8/8/8/8/8/2P5/4K3 w - - 0 1";
  const lczero::Move first_selected =
      ParseMove(MakeHistory(kFirstFen).Last(), "e2e3");
  const lczero::Move second_selected =
      ParseMove(MakeHistory(kSecondFen).Last(), "d2d3");
  const lczero::Move third_selected =
      ParseMove(MakeHistory(kThirdFen).Last(), "c2c3");

  ASSERT_TRUE(completed_game_queue.push(
      MakeSinglePlyCompletedGame(kFirstFen, "e2e3", lczero::GameResult::DRAW)));
  ASSERT_TRUE(completed_game_queue.push(MakeSinglePlyCompletedGame(
      kSecondFen, "d2d3", lczero::GameResult::DRAW)));
  ASSERT_TRUE(completed_game_queue.push(
      MakeSinglePlyCompletedGame(kThirdFen, "c2c3", lczero::GameResult::DRAW)));

  writer.Stop();

  const raw_chunk_format::ParsedChunk first_chunk =
      ReadChunk(temp_dir.path / raw_chunk_format::ChunkFileName(1));
  const raw_chunk_format::ParsedChunk second_chunk =
      ReadChunk(temp_dir.path / raw_chunk_format::ChunkFileName(2));
  const raw_chunk_format::ParsedChunk third_chunk =
      ReadChunk(temp_dir.path / raw_chunk_format::ChunkFileName(3));

  EXPECT_EQ(FirstSelectedMoveRaw(first_chunk), first_selected.raw_data());
  EXPECT_EQ(FirstSelectedMoveRaw(second_chunk), second_selected.raw_data());
  EXPECT_EQ(FirstSelectedMoveRaw(third_chunk), third_selected.raw_data());
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
