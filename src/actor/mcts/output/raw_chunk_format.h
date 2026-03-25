#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "chess/position.h"

namespace engine::raw_chunk_format {

inline constexpr std::array<char, 4> kChunkMagic = {'C', 'F', 'R', 'G'};
inline constexpr uint32_t kChunkVersion = 1;
// Absolute UCI like e2e4 or a7a8q.
inline constexpr std::size_t kStoredMoveUciBytes = 5;

struct StoredPolicyEntry {
  std::string move_uci;
  float prob = 0.0f;
};

struct StoredPly {
  std::string selected_move_uci;
  std::vector<StoredPolicyEntry> policy;
};

struct StoredRawGame {
  std::optional<std::string> initial_fen;
  lczero::GameResult game_result = lczero::GameResult::UNDECIDED;
  std::vector<StoredPly> plies;
};

struct ParsedChunk {
  uint32_t version = 0;
  std::vector<StoredRawGame> records;
};

std::string ChunkFileName(uint32_t chunk_index);

std::vector<char> SerializeChunkHeader();
std::vector<char> SerializeChunkRecord(const StoredRawGame& raw_game);

ParsedChunk ParseChunk(std::span<const uint8_t> bytes);

}  // namespace engine::raw_chunk_format
