#include "actor/mcts/runtime/match_game_runner.h"

#include <cassert>
#include <stdexcept>
#include <utility>

#include "chess/position.h"

namespace engine {
namespace {

std::string MoveToAbsoluteUci(const lczero::Position& position,
                              lczero::Move move) {
  if (position.GetBoard().flipped()) move.Flip();
  return move.ToString(false);
}

CompletedMatchGame BuildCompletedMatchGame(std::unique_ptr<GameTask> task,
                                           lczero::GameResult result) {
  assert(task != nullptr);
  assert(result != lczero::GameResult::UNDECIDED);

  MatchTask* match_task = dynamic_cast<MatchTask*>(task.get());
  if (match_task == nullptr) {
    throw std::runtime_error("match game runner expected MatchTask");
  }

  return CompletedMatchGame{
      .white_backend_slot = match_task->white_backend_slot,
      .black_backend_slot = match_task->black_backend_slot,
      .move_uci_history = std::move(match_task->move_uci_history),
      .game_result = result,
  };
}

}  // namespace

MatchGameRunner::MatchGameRunner(MatchGameRunnerChannels channels)
    : channels_(std::move(channels)) {
  assert(channels_.completion_queue != nullptr);
  assert(channels_.ready_queue != nullptr);
  assert(channels_.on_completed_game != nullptr);
}

MatchGameRunner::~MatchGameRunner() { Stop(); }

void MatchGameRunner::Start() {
  std::lock_guard<std::mutex> lock(state_mu_);
  if (started_ || stopped_) return;

  started_ = true;
  runner_ = std::thread(&MatchGameRunner::RunLoop, this);
}

void MatchGameRunner::Stop() {
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

void MatchGameRunner::RunLoop() {
  while (auto completed = channels_.completion_queue->pop()) {
    assert(completed->task != nullptr);
    assert(completed->task->searcher != nullptr);

    std::unique_ptr<GameTask> task = std::move(completed->task);
    MatchTask* match_task = dynamic_cast<MatchTask*>(task.get());
    if (match_task == nullptr) {
      throw std::runtime_error("match game runner expected MatchTask");
    }

    if (completed->result.game_result == lczero::GameResult::UNDECIDED) {
      assert(completed->result.root_history.GetLength() > 0);
      assert(match_task->inactive_searcher != nullptr);
      const lczero::Position& root_position =
          completed->result.root_history.Last();
      match_task->move_uci_history.push_back(
          MoveToAbsoluteUci(root_position, completed->result.selected_move));
      task->searcher->CommitMove(completed->result.selected_move);
      match_task->inactive_searcher->CommitMove(
          completed->result.selected_move);
      std::swap(task->searcher, match_task->inactive_searcher);
      match_task->active_player_is_white = !match_task->active_player_is_white;
      task->coroutine.reset();
      task->response.reset();
      task->state = TaskState::kNew;
      const bool pushed = channels_.ready_queue->push(std::move(task));
      assert(pushed);
      continue;
    }

    channels_.on_completed_game(BuildCompletedMatchGame(
        std::move(task), completed->result.game_result));
  }
}

}  // namespace engine
