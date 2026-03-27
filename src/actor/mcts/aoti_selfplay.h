#pragma once

#include <cstdint>
#include <filesystem>

namespace engine {

struct AotiSelfPlayOptions {
  std::filesystem::path model_package_path;
  int input_channels = 0;
  int policy_size = 0;
  std::filesystem::path raw_output_dir;
  int num_workers = 0;
  int num_games = 0;
  uint64_t raw_chunk_max_bytes = 0;
  int num_action = 0;
  int num_simulation = 0;
  float c_puct = 0.0f;
  float c_visit = 0.0f;
  float c_scale = 0.0f;
};

void RunAotiSelfPlay(const AotiSelfPlayOptions& options);

}  // namespace engine
