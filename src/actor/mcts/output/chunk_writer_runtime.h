#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <mutex>
#include <thread>

#include "actor/mcts/primitives/thread_safe_queue.h"
#include "actor/mcts/training_types.h"

namespace engine {

struct ChunkWriterChannels {
  ThreadSafeQueue<CompletedGame>* completed_game_queue = nullptr;
};

struct ChunkWriterOptions {
  static constexpr uint64_t kDefaultMaxChunkBytes = 128ull * 1024 * 1024;

  std::filesystem::path output_dir;
  uint64_t max_chunk_bytes = kDefaultMaxChunkBytes;
};

class ChunkWriterRuntime {
 public:
  ChunkWriterRuntime(ChunkWriterChannels channels, ChunkWriterOptions options);
  ~ChunkWriterRuntime();

  ChunkWriterRuntime(const ChunkWriterRuntime&) = delete;
  ChunkWriterRuntime& operator=(const ChunkWriterRuntime&) = delete;

  void Start();
  void Stop();

 private:
  void RunLoop();
  void OpenNextChunkFile(std::ofstream* stream);

  ChunkWriterChannels channels_;
  ChunkWriterOptions options_;
  uint32_t next_chunk_index_ = 1;
  uint64_t current_chunk_bytes_ = 0;
  uint32_t current_chunk_records_ = 0;
  bool started_ = false;
  bool stopped_ = false;
  mutable std::mutex state_mu_;
  std::thread writer_;
};

}  // namespace engine
