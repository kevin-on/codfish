#pragma once

#include <string>

#include "inference_types.h"
#include "model_manifest.h"
#include "status.h"

namespace engine {

class InferenceBackend {
 public:
  virtual ~InferenceBackend() = default;

  virtual Status Load(const ModelManifest& manifest) = 0;
  virtual Status Run(const InferenceBatch& batch, InferenceOutputs* out) = 0;
  virtual std::string Name() const = 0;
};

}  // namespace engine
