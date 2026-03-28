#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "actor/mcts/searcher.h"
#include "actor/mcts/training_types.h"

namespace engine {

enum class TaskState : uint8_t {
  kNew,
  kWaitingEval,
  kReady,
};

struct GameTask {
  virtual ~GameTask() = default;

  virtual int BackendSlotForRequestItem(const EvalRequestItem& item) const {
    static_cast<void>(item);
    return 0;
  }

  std::unique_ptr<MCTSSearcher> searcher;
  std::optional<SearchCoroutine> coroutine;
  TaskState state = TaskState::kNew;
  std::optional<EvalResponse> response;
  std::vector<TrainingSampleDraft> training_sample_drafts;
};

struct PendingEval {
  std::unique_ptr<GameTask> task;
  EvalRequest request;
};

struct CompletedSearch {
  std::unique_ptr<GameTask> task;
  SearchResult result;
};

}  // namespace engine
