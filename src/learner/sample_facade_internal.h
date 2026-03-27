#pragma once

#include <array>
#include <string>
#include <vector>

#include "engine/encoder.h"
#include "lc0/move_index.h"
#include "learner/raw_types.h"

namespace engine::learner::internal {

struct EncodedSampleDraft {
  std::array<uint8_t, kInputElements> input{};
  std::array<float, lczero::kPolicySize> policy_target{};
  std::array<float, 3> wdl_target{};
  std::string selected_move_uci;
};

std::vector<EncodedSampleDraft> BuildEncodedSampleDrafts(
    const RawGame& raw_game);

}  // namespace engine::learner::internal
