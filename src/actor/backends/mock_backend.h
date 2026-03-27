#pragma once

#include <random>

#include "engine/infer/inference_backend.h"

namespace engine {

class MockBackend final : public InferenceBackend {
 public:
  MockBackend() = default;
  ~MockBackend() override = default;

  Status Load() override;
  Status Run(const InferenceBatch& batch, InferenceOutputs* out) override;
  std::string Name() const override { return "mock"; }

 private:
  bool loaded_ = false;
  std::mt19937 rng_{std::random_device{}()};
};

}  // namespace engine
