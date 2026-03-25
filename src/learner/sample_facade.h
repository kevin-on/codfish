#pragma once

#include <cstdint>
#include <vector>

#include "engine/encoder.h"
#include "learner/raw_types.h"
#include "lc0/move_index.h"

namespace engine::learner {

struct EncodedGameSamples {
  int sample_count = 0;
  int input_channels = kInputPlanes;
  int policy_size = lczero::kPolicySize;
  std::vector<uint8_t> inputs;
  std::vector<float> policy_targets;
  std::vector<float> wdl_targets;
};

EncodedGameSamples EncodeRawGame(const RawGame& raw_game);

}  // namespace engine::learner
