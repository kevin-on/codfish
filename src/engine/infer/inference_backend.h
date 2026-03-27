#pragma once

#include <string>

#include "inference_types.h"
#include "status.h"

namespace engine {

class InferenceBackend {
 public:
  virtual ~InferenceBackend() = default;

  virtual Status Load() = 0;
  virtual Status Run(const InferenceBatch& batch, InferenceOutputs* out) = 0;
  virtual std::string Name() const = 0;
};

}  // namespace engine
