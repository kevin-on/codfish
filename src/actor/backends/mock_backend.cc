#include "mock_backend.h"

#include <algorithm>

namespace engine {

Status MockBackend::Load(const ModelManifest& manifest) {
  if (manifest.policy_size <= 0) {
    return Status::Error("invalid policy size");
  }
  if (manifest.input_channels <= 0) {
    return Status::Error("invalid input channels");
  }

  policy_size_ = manifest.policy_size;
  loaded_ = true;
  return Status::Ok();
}

Status MockBackend::Run(const InferenceBatch& batch, InferenceOutputs* out) {
  if (out == nullptr) return Status::Error("output buffer is null");

  out->policy_logits.clear();
  out->wdl_logits.clear();

  if (!loaded_) return Status::Error("mock backend not loaded");
  if (batch.batch_size < 0) return Status::Error("negative batch size");
  if (batch.batch_size > 0 && batch.planes == nullptr) {
    return Status::Error("planes buffer is null");
  }
  if (policy_size_ <= 0) return Status::Error("invalid policy size");

  const std::size_t batch_size = static_cast<std::size_t>(batch.batch_size);
  const std::size_t policy_size = static_cast<std::size_t>(policy_size_);

  out->policy_logits.resize(batch_size * policy_size);
  out->wdl_logits.resize(batch_size * 3);

  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::generate(out->policy_logits.begin(), out->policy_logits.end(),
                [this, &dist]() { return dist(rng_); });
  std::generate(out->wdl_logits.begin(), out->wdl_logits.end(),
                [this, &dist]() { return dist(rng_); });

  return Status::Ok();
}

}  // namespace engine
