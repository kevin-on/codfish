#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "actor/mcts/output/raw_chunk_format.h"

namespace {

engine::raw_chunk_format::StoredRawGame MakeCanonicalRawGame() {
  return engine::raw_chunk_format::StoredRawGame{
      .initial_fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
      .game_result = lczero::GameResult::WHITE_WON,
      .plies =
          {
              engine::raw_chunk_format::StoredPly{
                  .selected_move_uci = "e2e4",
                  .policy =
                      {
                          engine::raw_chunk_format::StoredPolicyEntry{
                              .move_uci = "e2e4",
                              .prob = 0.75f,
                          },
                          engine::raw_chunk_format::StoredPolicyEntry{
                              .move_uci = "e2e3",
                              .prob = 0.25f,
                          },
                      },
              },
          },
  };
}

int WriteCanonicalChunk(const std::filesystem::path& output_path) {
  const std::filesystem::path parent_path = output_path.parent_path();
  if (!parent_path.empty()) {
    std::filesystem::create_directories(parent_path);
  }

  std::ofstream stream(output_path,
                       std::ios::binary | std::ios::out | std::ios::trunc);
  if (!stream.is_open()) {
    std::cerr << "failed to open output: " << output_path << "\n";
    return 1;
  }

  const std::vector<char> header = engine::raw_chunk_format::SerializeChunkHeader();
  const std::vector<char> record =
      engine::raw_chunk_format::SerializeChunkRecord(MakeCanonicalRawGame());

  stream.write(header.data(), static_cast<std::streamsize>(header.size()));
  stream.write(record.data(), static_cast<std::streamsize>(record.size()));
  if (!stream.good()) {
    std::cerr << "failed to write canonical chunk: " << output_path << "\n";
    return 1;
  }

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: learner_raw_chunk_golden_writer <output_path>\n";
    return 2;
  }

  return WriteCanonicalChunk(argv[1]);
}
