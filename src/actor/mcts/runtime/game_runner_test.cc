#include "actor/mcts/runtime/game_runner.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <utility>

namespace engine {
namespace {

class CommitRecordingSearcher final : public MCTSSearcher {
 public:
  SearchCoroutine Run() override {
    SearchResult result;
    result.game_result = lczero::GameResult::DRAW;
    co_return result;
  }

  void CommitMove(lczero::Move move) override {
    ++commit_calls_;
    committed_move_ = move;
  }

  int commit_calls() const { return commit_calls_; }
  lczero::Move committed_move() const { return committed_move_; }

 private:
  int commit_calls_ = 0;
  lczero::Move committed_move_;
};

lczero::PositionHistory MakeStartHistory() {
  lczero::PositionHistory history;
  history.Reset(lczero::Position::FromFen(lczero::ChessBoard::kStartposFen));
  return history;
}

lczero::Move ParseMove(const char* uci) {
  const lczero::Position start =
      lczero::Position::FromFen(lczero::ChessBoard::kStartposFen);
  return start.GetBoard().ParseMove(uci);
}

std::unique_ptr<GameTask> MakeTask(std::unique_ptr<MCTSSearcher> searcher,
                                   TaskState state) {
  auto task = std::make_unique<GameTask>();
  task->searcher = std::move(searcher);
  task->state = state;
  return task;
}

TEST(GameRunner, NonTerminalResultAppendsDraftCommitsMoveAndRequeuesTask) {
  ThreadSafeQueue<CompletedSearch> completion_queue;
  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue;
  ThreadSafeQueue<CompletedGame> completed_game_queue;
  GameRunner runner(GameRunnerChannels{
      .completion_queue = &completion_queue,
      .ready_queue = &ready_queue,
      .completed_game_queue = &completed_game_queue,
  });
  runner.Start();

  auto searcher = std::make_unique<CommitRecordingSearcher>();
  CommitRecordingSearcher* searcher_raw = searcher.get();
  auto task = MakeTask(std::move(searcher), TaskState::kReady);
  task->coroutine.emplace(task->searcher->Run());
  task->response.emplace();

  const lczero::Move move = ParseMove("e2e4");
  SearchResult result;
  result.root_history = MakeStartHistory();
  result.selected_move = move;
  result.legal_moves = {move};
  result.improved_policy = {1.0f};

  ASSERT_TRUE(completion_queue.push(
      CompletedSearch{.task = std::move(task), .result = std::move(result)}));

  std::optional<std::unique_ptr<GameTask>> requeued = ready_queue.pop();
  ASSERT_TRUE(requeued.has_value());
  ASSERT_TRUE(*requeued != nullptr);
  EXPECT_EQ(searcher_raw->commit_calls(), 1);
  EXPECT_EQ(searcher_raw->committed_move(), move);
  EXPECT_EQ((*requeued)->state, TaskState::kNew);
  EXPECT_FALSE((*requeued)->coroutine.has_value());
  EXPECT_FALSE((*requeued)->response.has_value());
  ASSERT_EQ((*requeued)->training_sample_drafts.size(), 1u);
  EXPECT_EQ((*requeued)->training_sample_drafts[0].root_history.GetLength(), 1);
  EXPECT_EQ((*requeued)->training_sample_drafts[0].selected_move, move);
  ASSERT_EQ((*requeued)->training_sample_drafts[0].legal_moves.size(), 1u);
  EXPECT_EQ((*requeued)->training_sample_drafts[0].legal_moves[0], move);
  ASSERT_EQ((*requeued)->training_sample_drafts[0].improved_policy.size(), 1u);
  EXPECT_FLOAT_EQ((*requeued)->training_sample_drafts[0].improved_policy[0],
                  1.0f);

  completed_game_queue.close();
  ready_queue.close();
  runner.Stop();
}

TEST(GameRunner, TerminalResultBuildsCompletedGameAndHandsItOff) {
  ThreadSafeQueue<CompletedSearch> completion_queue;
  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue;
  ThreadSafeQueue<CompletedGame> completed_game_queue;
  std::atomic<int> completed_callbacks{0};
  GameRunner runner(GameRunnerChannels{
      .completion_queue = &completion_queue,
      .ready_queue = &ready_queue,
      .completed_game_queue = &completed_game_queue,
      .on_completed_game = [&completed_callbacks] { ++completed_callbacks; },
  });
  runner.Start();

  auto task =
      MakeTask(std::make_unique<CommitRecordingSearcher>(), TaskState::kNew);

  const lczero::Move move = ParseMove("d2d4");
  task->training_sample_drafts.push_back(TrainingSampleDraft{
      .root_history = MakeStartHistory(),
      .selected_move = move,
      .legal_moves = {move},
      .improved_policy = {0.75f},
  });

  SearchResult terminal;
  terminal.game_result = lczero::GameResult::WHITE_WON;
  ASSERT_TRUE(completion_queue.push(CompletedSearch{
      .task = std::move(task),
      .result = std::move(terminal),
  }));

  std::optional<CompletedGame> completed_game = completed_game_queue.pop();
  ASSERT_TRUE(completed_game.has_value());
  EXPECT_EQ(completed_callbacks.load(), 1);
  EXPECT_EQ(completed_game->game_result, lczero::GameResult::WHITE_WON);
  ASSERT_EQ(completed_game->sample_drafts.size(), 1u);
  EXPECT_EQ(completed_game->sample_drafts[0].root_history.GetLength(), 1);
  EXPECT_EQ(completed_game->sample_drafts[0].selected_move, move);
  ASSERT_EQ(completed_game->sample_drafts[0].legal_moves.size(), 1u);
  EXPECT_EQ(completed_game->sample_drafts[0].legal_moves[0], move);
  ASSERT_EQ(completed_game->sample_drafts[0].improved_policy.size(), 1u);
  EXPECT_FLOAT_EQ(completed_game->sample_drafts[0].improved_policy[0], 0.75f);
  EXPECT_FALSE(ready_queue
                   .pop_until(std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(50))
                   .has_value());

  ready_queue.close();
  runner.Stop();
}

TEST(GameRunner, StopWakesBlockedRunner) {
  ThreadSafeQueue<CompletedSearch> completion_queue;
  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue;
  ThreadSafeQueue<CompletedGame> completed_game_queue;
  GameRunner runner(GameRunnerChannels{
      .completion_queue = &completion_queue,
      .ready_queue = &ready_queue,
      .completed_game_queue = &completed_game_queue,
  });
  runner.Start();

  std::future<void> stop_future =
      std::async(std::launch::async, [&runner] { runner.Stop(); });
  EXPECT_EQ(stop_future.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  stop_future.get();

  ready_queue.close();
  completed_game_queue.close();
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
