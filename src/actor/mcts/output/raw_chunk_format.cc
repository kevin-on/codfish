#include "actor/mcts/output/raw_chunk_format.h"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace engine::raw_chunk_format {
namespace {

uint32_t ToUint32(std::size_t value) {
  assert(value <= std::numeric_limits<uint32_t>::max());
  return static_cast<uint32_t>(value);
}

void AppendUint8(std::vector<char>* buffer, uint8_t value) {
  buffer->push_back(static_cast<char>(value));
}

void AppendUint16(std::vector<char>* buffer, uint16_t value) {
  buffer->push_back(static_cast<char>(value & 0xff));
  buffer->push_back(static_cast<char>((value >> 8) & 0xff));
}

void AppendUint32(std::vector<char>* buffer, uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    buffer->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendFloat32(std::vector<char>* buffer, float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  AppendUint32(buffer, bits);
}

void AppendBytes(std::vector<char>* buffer, std::string_view value) {
  buffer->insert(buffer->end(), value.begin(), value.end());
}

void AppendStoredMoveUci(std::vector<char>* buffer, std::string_view uci) {
  if (uci.size() < 4 || uci.size() > kStoredMoveUciBytes) {
    throw std::runtime_error("bad stored move uci length");
  }
  buffer->insert(buffer->end(), uci.begin(), uci.end());
  for (std::size_t i = uci.size(); i < kStoredMoveUciBytes; ++i) {
    buffer->push_back('\0');
  }
}

void RequireRemaining(std::span<const uint8_t> bytes, std::size_t offset,
                      std::size_t need) {
  if (offset + need > bytes.size()) {
    throw std::runtime_error("truncated raw chunk");
  }
}

uint8_t ReadUint8(std::span<const uint8_t> bytes, std::size_t* offset) {
  RequireRemaining(bytes, *offset, 1);
  return bytes[(*offset)++];
}

uint16_t ReadUint16(std::span<const uint8_t> bytes, std::size_t* offset) {
  RequireRemaining(bytes, *offset, 2);
  const uint16_t value = static_cast<uint16_t>(bytes[*offset]) |
                         (static_cast<uint16_t>(bytes[*offset + 1]) << 8);
  *offset += 2;
  return value;
}

uint32_t ReadUint32(std::span<const uint8_t> bytes, std::size_t* offset) {
  RequireRemaining(bytes, *offset, 4);
  uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8) {
    value |= static_cast<uint32_t>(bytes[*offset + shift / 8]) << shift;
  }
  *offset += 4;
  return value;
}

float ReadFloat32(std::span<const uint8_t> bytes, std::size_t* offset) {
  const uint32_t bits = ReadUint32(bytes, offset);
  float value = 0.0f;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::string ReadString(std::span<const uint8_t> bytes, std::size_t* offset,
                       std::size_t len) {
  RequireRemaining(bytes, *offset, len);
  const char* begin = reinterpret_cast<const char*>(bytes.data() + *offset);
  std::string value(begin, len);
  *offset += len;
  return value;
}

std::string ReadStoredMoveUci(std::span<const uint8_t> bytes,
                              std::size_t* offset) {
  RequireRemaining(bytes, *offset, kStoredMoveUciBytes);
  const char* begin = reinterpret_cast<const char*>(bytes.data() + *offset);
  const std::string_view encoded(begin, kStoredMoveUciBytes);
  *offset += kStoredMoveUciBytes;

  if (encoded[0] == '\0' || encoded[1] == '\0' || encoded[2] == '\0' ||
      encoded[3] == '\0') {
    throw std::runtime_error("bad stored move uci");
  }
  if (encoded[4] == '\0') {
    return std::string(encoded.substr(0, 4));
  }
  return std::string(encoded);
}

void SerializeStoredPly(const StoredPly& ply, std::vector<char>* record) {
  AppendStoredMoveUci(record, ply.selected_move_uci);
  AppendUint32(record, ToUint32(ply.policy.size()));
  for (const StoredPolicyEntry& entry : ply.policy) {
    AppendStoredMoveUci(record, entry.move_uci);
    AppendFloat32(record, entry.prob);
  }
}

StoredRawGame ParseStoredRawGameRecord(std::span<const uint8_t> bytes) {
  std::size_t offset = 0;

  StoredRawGame raw_game;
  const bool has_initial_fen = ReadUint8(bytes, &offset) != 0;
  raw_game.game_result =
      static_cast<lczero::GameResult>(ReadUint8(bytes, &offset));
  const uint32_t initial_fen_len = ReadUint32(bytes, &offset);
  if (has_initial_fen) {
    raw_game.initial_fen = ReadString(bytes, &offset, initial_fen_len);
  } else if (initial_fen_len != 0) {
    throw std::runtime_error("raw chunk missing initial fen flag");
  }

  const uint32_t num_plies = ReadUint32(bytes, &offset);
  raw_game.plies.reserve(num_plies);
  for (uint32_t ply_idx = 0; ply_idx < num_plies; ++ply_idx) {
    StoredPly ply;
    ply.selected_move_uci = ReadStoredMoveUci(bytes, &offset);

    const uint32_t policy_size = ReadUint32(bytes, &offset);
    ply.policy.reserve(policy_size);
    for (uint32_t policy_idx = 0; policy_idx < policy_size; ++policy_idx) {
      ply.policy.push_back(StoredPolicyEntry{
          .move_uci = ReadStoredMoveUci(bytes, &offset),
          .prob = ReadFloat32(bytes, &offset),
      });
    }
    raw_game.plies.push_back(std::move(ply));
  }

  if (offset != bytes.size()) {
    throw std::runtime_error("raw chunk record length mismatch");
  }
  return raw_game;
}

std::vector<char> SerializeStoredRawGamePayload(const StoredRawGame& raw_game) {
  std::vector<char> payload;
  AppendUint8(&payload, raw_game.initial_fen.has_value() ? 1 : 0);
  AppendUint8(&payload, static_cast<uint8_t>(raw_game.game_result));
  AppendUint32(&payload, raw_game.initial_fen.has_value()
                             ? ToUint32(raw_game.initial_fen->size())
                             : 0);
  if (raw_game.initial_fen.has_value()) {
    AppendBytes(&payload, *raw_game.initial_fen);
  }

  AppendUint32(&payload, ToUint32(raw_game.plies.size()));
  for (const StoredPly& ply : raw_game.plies) {
    SerializeStoredPly(ply, &payload);
  }
  return payload;
}

}  // namespace

std::string ChunkFileName(uint32_t chunk_index) {
  assert(chunk_index > 0);
  char file_name[32];
  const int written = std::snprintf(file_name, sizeof(file_name),
                                    "games-%06u.bin", chunk_index);
  assert(written > 0);
  return std::string(file_name);
}

std::vector<char> SerializeChunkHeader() {
  std::vector<char> header;
  header.insert(header.end(), kChunkMagic.begin(), kChunkMagic.end());
  AppendUint32(&header, kChunkVersion);
  return header;
}

std::vector<char> SerializeChunkRecord(const StoredRawGame& raw_game) {
  const std::vector<char> payload = SerializeStoredRawGamePayload(raw_game);

  std::vector<char> record;
  record.reserve(sizeof(uint32_t) + payload.size());
  AppendUint32(&record, ToUint32(payload.size()));
  record.insert(record.end(), payload.begin(), payload.end());
  return record;
}

ParsedChunk ParseChunk(std::span<const uint8_t> bytes) {
  std::size_t offset = 0;

  RequireRemaining(bytes, offset, kChunkMagic.size());
  for (char expected : kChunkMagic) {
    if (static_cast<char>(ReadUint8(bytes, &offset)) != expected) {
      throw std::runtime_error("bad raw chunk magic");
    }
  }

  ParsedChunk chunk;
  chunk.version = ReadUint32(bytes, &offset);
  if (chunk.version != kChunkVersion) {
    throw std::runtime_error("unsupported raw chunk version");
  }
  while (offset < bytes.size()) {
    const uint32_t record_len = ReadUint32(bytes, &offset);
    RequireRemaining(bytes, offset, record_len);

    const std::span<const uint8_t> record_bytes =
        bytes.subspan(offset, record_len);
    chunk.records.push_back(ParseStoredRawGameRecord(record_bytes));
    offset += record_len;
  }

  return chunk;
}

}  // namespace engine::raw_chunk_format
