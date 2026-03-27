#include "aoti_backend.h"

#include <ATen/Tensor.h>
#include <torch/torch.h>

#include <exception>
#include <vector>

namespace engine {

Status AotiBackend::Load() {
  if (input_channels_ <= 0) {
    return Status::Error("invalid input channels");
  }
  if (policy_size_ <= 0) {
    return Status::Error("invalid policy size");
  }
  if (model_package_path_.empty()) {
    return Status::Error("model package path must not be empty");
  }

  try {
    loader_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        model_package_path_.string());
  } catch (const std::exception& exc) {
    return Status::Error(exc.what());
  }

  loaded_ = true;
  return Status::Ok();
}

Status AotiBackend::Run(const InferenceBatch& batch, InferenceOutputs* out) {
  if (out == nullptr) return Status::Error("output buffer is null");

  out->policy_logits.clear();
  out->wdl_probs.clear();

  if (!loaded_) return Status::Error("AOTI backend not loaded");
  if (batch.batch_size < 0) return Status::Error("negative batch size");
  if (batch.batch_size > 0 && batch.planes == nullptr) {
    return Status::Error("planes buffer is null");
  }

  try {
    const auto batch_size = static_cast<int64_t>(batch.batch_size);
    const auto input_channels = static_cast<int64_t>(input_channels_);
    auto cpu_input = torch::from_blob(
        const_cast<void*>(batch.planes), {batch_size, input_channels, 8, 8},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
    auto input = cpu_input.to(
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32),
        /*non_blocking=*/false, /*copy=*/true);

    std::vector<at::Tensor> outputs = loader_->run({input});
    if (outputs.size() != 2) {
      return Status::Error("AOTI model must return exactly two outputs");
    }

    at::Tensor policy = outputs[0].contiguous().to(torch::kCPU);
    at::Tensor wdl = outputs[1].contiguous().to(torch::kCPU);
    if (policy.dim() != 2 || policy.size(0) != batch_size ||
        policy.size(1) != policy_size_) {
      return Status::Error("unexpected policy output shape");
    }
    if (wdl.dim() != 2 || wdl.size(0) != batch_size || wdl.size(1) != 3) {
      return Status::Error("unexpected WDL output shape");
    }

    const float* policy_ptr = policy.const_data_ptr<float>();
    const float* wdl_ptr = wdl.const_data_ptr<float>();
    out->policy_logits.assign(
        policy_ptr, policy_ptr + static_cast<std::size_t>(policy.numel()));
    out->wdl_probs.assign(wdl_ptr,
                          wdl_ptr + static_cast<std::size_t>(wdl.numel()));
  } catch (const std::exception& exc) {
    return Status::Error(exc.what());
  }

  return Status::Ok();
}

}  // namespace engine
