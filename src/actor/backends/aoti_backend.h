#pragma once

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <filesystem>
#include <memory>
#include <utility>

#include "engine/infer/inference_backend.h"

namespace engine {

class AotiBackend final : public InferenceBackend {
 public:
  AotiBackend(std::filesystem::path model_package_path, int input_channels,
              int policy_size)
      : model_package_path_(std::move(model_package_path)),
        input_channels_(input_channels),
        policy_size_(policy_size) {}
  ~AotiBackend() override = default;

  Status Load() override;
  Status Run(const InferenceBatch& batch, InferenceOutputs* out) override;
  std::string Name() const override { return "aoti"; }

 private:
  std::filesystem::path model_package_path_;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
  int input_channels_ = 0;
  int policy_size_ = 0;
  bool loaded_ = false;
};

}  // namespace engine
