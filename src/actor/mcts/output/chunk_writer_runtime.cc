#include "actor/mcts/output/chunk_writer_runtime.h"

#include <cassert>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "chess/position.h"
#include "actor/mcts/output/raw_chunk_format.h"

namespace engine {
namespace {

std::optional<std::string> ExtractInitialFen(const CompletedGame& completed) {
  if (completed.sample_drafts.empty()) return std::nullopt;
  if (completed.sample_drafts.front().root_history.GetLength() == 0) {
    return std::nullopt;
  }
  return lczero::PositionToFen(
      completed.sample_drafts.front().root_history.Starting());
}

raw_chunk_format::StoredRawGame BuildStoredRawGame(
    const CompletedGame& completed) {
  assert(completed.game_result != lczero::GameResult::UNDECIDED);

  raw_chunk_format::StoredRawGame raw_game;
  raw_game.initial_fen = ExtractInitialFen(completed);
  raw_game.game_result = completed.game_result;
  raw_game.plies.reserve(completed.sample_drafts.size());
  for (const TrainingSampleDraft& draft : completed.sample_drafts) {
    assert(draft.legal_moves.size() == draft.improved_policy.size());

    raw_chunk_format::StoredPly ply;
    ply.selected_move_raw = draft.selected_move.raw_data();
    ply.policy.reserve(draft.legal_moves.size());
    for (std::size_t i = 0; i < draft.legal_moves.size(); ++i) {
      ply.policy.push_back(raw_chunk_format::StoredPolicyEntry{
          .move_raw = draft.legal_moves[i].raw_data(),
          .prob = draft.improved_policy[i],
      });
    }
    raw_game.plies.push_back(std::move(ply));
  }

  return raw_game;
}

uint64_t RecordWriteSize(const std::vector<char>& record) {
  return static_cast<uint64_t>(record.size());
}

void FlushChunkFile(std::ofstream* stream) {
  stream->flush();
  assert(stream->good());
}

void CloseChunkFile(std::ofstream* stream) {
  if (!stream->is_open()) return;
  FlushChunkFile(stream);
  stream->close();
}

void OpenChunkFile(const ChunkWriterOptions& options, uint32_t chunk_index,
                   uint64_t* current_chunk_bytes,
                   uint32_t* current_chunk_records, std::ofstream* stream) {
  std::filesystem::create_directories(options.output_dir);
  stream->open(
      options.output_dir /
          std::filesystem::path(raw_chunk_format::ChunkFileName(chunk_index)),
      std::ios::binary | std::ios::out | std::ios::trunc);
  assert(stream->is_open());

  const std::vector<char> header = raw_chunk_format::SerializeChunkHeader();
  stream->write(header.data(), static_cast<std::streamsize>(header.size()));
  assert(stream->good());
  *current_chunk_bytes = header.size();
  *current_chunk_records = 0;
}

void AppendRecord(const std::vector<char>& record,
                  uint64_t* current_chunk_bytes,
                  uint32_t* current_chunk_records, std::ofstream* stream) {
  stream->write(record.data(), static_cast<std::streamsize>(record.size()));
  FlushChunkFile(stream);
  *current_chunk_bytes += RecordWriteSize(record);
  ++(*current_chunk_records);
}

}  // namespace

ChunkWriterRuntime::ChunkWriterRuntime(ChunkWriterChannels channels,
                                       ChunkWriterOptions options)
    : channels_(channels), options_(std::move(options)) {
  assert(channels_.completed_game_queue != nullptr);
  assert(!options_.output_dir.empty());
  assert(options_.max_chunk_bytes > 0);
}

ChunkWriterRuntime::~ChunkWriterRuntime() { Stop(); }

void ChunkWriterRuntime::Start() {
  std::lock_guard<std::mutex> lock(state_mu_);
  if (started_) return;
  if (stopped_) return;

  started_ = true;
  writer_ = std::thread(&ChunkWriterRuntime::RunLoop, this);
}

void ChunkWriterRuntime::Stop() {
  bool should_stop = false;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (!started_ || stopped_) return;
    stopped_ = true;
    should_stop = true;
  }
  if (!should_stop) return;

  channels_.completed_game_queue->close();
  if (writer_.joinable()) writer_.join();
}

void ChunkWriterRuntime::OpenNextChunkFile(std::ofstream* stream) {
  OpenChunkFile(options_, next_chunk_index_, &current_chunk_bytes_,
                &current_chunk_records_, stream);
  ++next_chunk_index_;
}

void ChunkWriterRuntime::RunLoop() {
  std::ofstream chunk_file;
  while (auto completed = channels_.completed_game_queue->pop()) {
    const raw_chunk_format::StoredRawGame raw_game =
        BuildStoredRawGame(*completed);
    const std::vector<char> record =
        raw_chunk_format::SerializeChunkRecord(raw_game);

    if (!chunk_file.is_open()) {
      OpenNextChunkFile(&chunk_file);
    } else if (current_chunk_records_ > 0 &&
               current_chunk_bytes_ + RecordWriteSize(record) >
                   options_.max_chunk_bytes) {
      CloseChunkFile(&chunk_file);
      OpenNextChunkFile(&chunk_file);
    }
    AppendRecord(record, &current_chunk_bytes_, &current_chunk_records_,
                 &chunk_file);
  }

  CloseChunkFile(&chunk_file);
}

}  // namespace engine
