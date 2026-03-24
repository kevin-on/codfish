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
  //   wdl_probs:     B * 3 normalized W/D/L probabilities in
  //                  win/draw/loss order, summing to 1
  std::vector<float> policy_logits;
  std::vector<float> wdl_probs;
};

}  // namespace engine
