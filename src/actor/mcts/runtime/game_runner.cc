#include "actor/mcts/runtime/game_runner.h"

#include <cassert>
#include <utility>

namespace engine {
namespace {

CompletedGame BuildCompletedGame(std::unique_ptr<GameTask> task,
                                 lczero::GameResult result) {
  assert(task != nullptr);
  assert(result != lczero::GameResult::UNDECIDED);

  return CompletedGame{
      .sample_drafts = std::move(task->training_sample_drafts),
      .game_result = result,
  };
}

}  // namespace

GameRunner::GameRunner(GameRunnerChannels channels) : channels_(channels) {
  assert(channels_.completion_queue != nullptr);
  assert(channels_.ready_queue != nullptr);
  assert(channels_.completed_game_queue != nullptr);
}

GameRunner::~GameRunner() { Stop(); }

void GameRunner::Start() {
  std::lock_guard<std::mutex> lock(state_mu_);
  if (started_) return;
  if (stopped_) return;

  started_ = true;
  runner_ = std::thread(&GameRunner::RunLoop, this);
}

void GameRunner::Stop() {
  bool should_stop = false;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (!started_ || stopped_) return;
    stopped_ = true;
    should_stop = true;
  }
  if (!should_stop) return;

  channels_.completion_queue->close();
  if (runner_.joinable()) runner_.join();
}

void GameRunner::RunLoop() {
  while (auto completed = channels_.completion_queue->pop()) {
    assert(completed->task != nullptr);
    assert(completed->task->searcher != nullptr);

    std::unique_ptr<GameTask> task = std::move(completed->task);
    if (completed->result.game_result == lczero::GameResult::UNDECIDED) {
      assert(completed->result.root_history.GetLength() > 0);
      assert(completed->result.legal_moves.size() ==
             completed->result.improved_policy.size());
      task->training_sample_drafts.push_back(TrainingSampleDraft{
          .root_history = std::move(completed->result.root_history),
          .selected_move = completed->result.selected_move,
          .legal_moves = std::move(completed->result.legal_moves),
          .improved_policy = std::move(completed->result.improved_policy),
      });
      task->searcher->CommitMove(completed->result.selected_move);
      task->coroutine.reset();
      task->response.reset();
      task->state = TaskState::kNew;
      const bool pushed = channels_.ready_queue->push(std::move(task));
      assert(pushed);
      continue;
    }

    CompletedGame completed_game =
        BuildCompletedGame(std::move(task), completed->result.game_result);
    const bool pushed =
        channels_.completed_game_queue->push(std::move(completed_game));
    assert(pushed);
    if (channels_.on_completed_game) {
      channels_.on_completed_game();
    }
  }
}

}  // namespace engine
