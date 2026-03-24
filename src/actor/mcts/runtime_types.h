#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "searcher.h"

namespace engine {

enum class TaskState : uint8_t {
  kNew,
  kWaitingEval,
  kReady,
};

struct TrainingSampleDraft {
  lczero::PositionHistory root_history;
  std::vector<lczero::Move> legal_moves;
  std::vector<float> improved_policy;
};

struct CompletedGame {
  std::vector<TrainingSampleDraft> sample_drafts;
  lczero::GameResult game_result = lczero::GameResult::UNDECIDED;
};

struct GameTask {
  virtual ~GameTask() = default;

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
