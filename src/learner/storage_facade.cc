#include "learner/storage_facade.h"

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

#include "actor/mcts/output/raw_chunk_format.h"

namespace engine::learner {
namespace {

std::vector<uint8_t> ReadFileBytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream.is_open()) {
    throw std::runtime_error("failed to open raw chunk file");
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream),
                              std::istreambuf_iterator<char>());
}

RawPolicyEntry ToRawPolicyEntry(raw_chunk_format::StoredPolicyEntry& entry) {
  return RawPolicyEntry{
      .move_uci = std::move(entry.move_uci),
      .prob = entry.prob,
  };
}

RawPly ToRawPly(raw_chunk_format::StoredPly& ply) {
  RawPly raw_ply;
  raw_ply.selected_move_uci = std::move(ply.selected_move_uci);
  raw_ply.policy.reserve(ply.policy.size());
  for (raw_chunk_format::StoredPolicyEntry& entry : ply.policy) {
    raw_ply.policy.push_back(ToRawPolicyEntry(entry));
  }
  return raw_ply;
}

RawGame ToRawGame(raw_chunk_format::StoredRawGame& game) {
  RawGame raw_game;
  raw_game.initial_fen = std::move(game.initial_fen);
  raw_game.game_result = game.game_result;
  raw_game.plies.reserve(game.plies.size());
  for (raw_chunk_format::StoredPly& ply : game.plies) {
    raw_game.plies.push_back(ToRawPly(ply));
  }
  return raw_game;
}

}  // namespace

RawChunkFile ReadRawChunkFile(const std::filesystem::path& path) {
  raw_chunk_format::ParsedChunk parsed =
      raw_chunk_format::ParseChunk(ReadFileBytes(path));

  RawChunkFile chunk;
  chunk.version = parsed.version;
  chunk.games.reserve(parsed.records.size());
  for (raw_chunk_format::StoredRawGame& game : parsed.records) {
    chunk.games.push_back(ToRawGame(game));
  }
  return chunk;
}

}  // namespace engine::learner
