#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

#include "actor/mcts/primitives/thread_safe_queue.h"
#include "actor/mcts/runtime/task_types.h"
#include "engine/encoder.h"
#include "engine/infer/inference_backend.h"
#include "engine/infer/model_manifest.h"

namespace engine {

struct InferenceChannels {
  ThreadSafeQueue<PendingEval>* request_queue = nullptr;
  ThreadSafeQueue<std::unique_ptr<GameTask>>* ready_queue = nullptr;
};

struct InferenceRuntimeOptions {
  int max_batch_size = 0;
  std::chrono::microseconds flush_timeout{0};
};

class InferenceRuntime {
 public:
  InferenceRuntime(InferenceChannels channels,
                   std::shared_ptr<InferenceBackend> backend,
                   const FeatureEncoder* encoder, ModelManifest manifest,
                   InferenceRuntimeOptions options);
  ~InferenceRuntime();

  InferenceRuntime(const InferenceRuntime&) = delete;
  InferenceRuntime& operator=(const InferenceRuntime&) = delete;

  void Start();
  void Stop();

 private:
  void RunLoop();

  InferenceChannels channels_;
  std::shared_ptr<InferenceBackend> backend_;
  const FeatureEncoder* encoder_ = nullptr;
  ModelManifest manifest_;
  InferenceRuntimeOptions options_;
  bool started_ = false;
  bool stopped_ = false;
  mutable std::mutex state_mu_;
  std::thread inference_thread_;
};

}  // namespace engine
