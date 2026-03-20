#pragma once

#include "engine/encoder.h"
#include "move_index.h"

namespace engine {

struct ModelManifest {
  int input_channels = kInputPlanes;
  int policy_size = lczero::kPolicySize;
};

}  // namespace engine
