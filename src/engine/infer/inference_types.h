#pragma once

#include <vector>

namespace engine {

struct InferenceBatch {
  // Contiguous [B, C, 8, 8] planes buffer owned by the caller.
  const void* planes = nullptr;
  int batch_size = 0;
};

struct InferenceOutputs {
  // Flattened per-batch outputs:
  //   policy_logits: B * policy_size
  //   wdl_logits:    B * 3
  std::vector<float> policy_logits;
  std::vector<float> wdl_logits;
};

}  // namespace engine
