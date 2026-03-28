#pragma once

#include <array>
#include <filesystem>
#include <string>

namespace engine {

struct AotiMatchOptions {
  std::array<std::filesystem::path, 2> model_package_paths;
  std::array<std::string, 2> player_names;
  int input_channels = 0;
  int policy_size = 0;
  std::filesystem::path output_pgn_path;
  int num_workers = 0;
  int num_games = 0;
  int num_action = 0;
  int num_simulation = 0;
  float c_puct = 0.0f;
  float c_visit = 0.0f;
  float c_scale = 0.0f;
};

void RunAotiMatch(const AotiMatchOptions& options);

}  // namespace engine
