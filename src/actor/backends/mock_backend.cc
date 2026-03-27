#include "mock_backend.h"

#include <algorithm>

#include "lc0/move_index.h"

namespace engine {

Status MockBackend::Load() {
  loaded_ = true;
  return Status::Ok();
}

Status MockBackend::Run(const InferenceBatch& batch, InferenceOutputs* out) {
  if (out == nullptr) return Status::Error("output buffer is null");

  out->policy_logits.clear();
  out->wdl_probs.clear();

  if (!loaded_) return Status::Error("mock backend not loaded");
  if (batch.batch_size < 0) return Status::Error("negative batch size");
  if (batch.batch_size > 0 && batch.planes == nullptr) {
    return Status::Error("planes buffer is null");
  }

  const std::size_t batch_size = static_cast<std::size_t>(batch.batch_size);
  const std::size_t policy_size = static_cast<std::size_t>(lczero::kPolicySize);

  out->policy_logits.resize(batch_size * policy_size);
  out->wdl_probs.resize(batch_size * 3);

  std::uniform_real_distribution<float> logit_dist(-1.0f, 1.0f);
  std::generate(out->policy_logits.begin(), out->policy_logits.end(),
                [this, &logit_dist]() { return logit_dist(rng_); });

  std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
  for (std::size_t b = 0; b < batch_size; ++b) {
    float w = prob_dist(rng_);
    float d = prob_dist(rng_);
    float l = prob_dist(rng_);
    const float sum = w + d + l;
    const float inv = 1.0f / sum;
    out->wdl_probs[b * 3 + 0] = w * inv;
    out->wdl_probs[b * 3 + 1] = d * inv;
    out->wdl_probs[b * 3 + 2] = l * inv;
  }

  return Status::Ok();
}

}  // namespace engine
