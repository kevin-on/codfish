#include "actor/mcts/runtime/match_game_runner.h"

#include <gtest/gtest.h>

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

std::unique_ptr<GameTask> MakeMatchTask(std::unique_ptr<MCTSSearcher> searcher,
                                        TaskState state, int white_backend_slot,
                                        int black_backend_slot) {
  auto task = std::make_unique<MatchTask>();
  task->searcher = std::move(searcher);
  task->state = state;
  task->white_backend_slot = white_backend_slot;
  task->black_backend_slot = black_backend_slot;
  return task;
}

TEST(MatchGameRunner, NonTerminalResultCommitsMoveAndRequeuesMatchTask) {
  ThreadSafeQueue<CompletedSearch> completion_queue;
  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue;
  std::promise<CompletedMatchGame> completed_promise;
  auto completed_future = completed_promise.get_future();
  MatchGameRunner runner(MatchGameRunnerChannels{
      .completion_queue = &completion_queue,
      .ready_queue = &ready_queue,
      .on_completed_game =
          [&completed_promise](CompletedMatchGame completed_game) {
            completed_promise.set_value(std::move(completed_game));
          },
  });
  runner.Start();

  auto searcher = std::make_unique<CommitRecordingSearcher>();
  CommitRecordingSearcher* searcher_raw = searcher.get();
  auto task = MakeMatchTask(std::move(searcher), TaskState::kReady,
                            /*white_backend_slot=*/0,
                            /*black_backend_slot=*/1);
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
  auto* match_task = dynamic_cast<MatchTask*>(requeued->get());
  ASSERT_NE(match_task, nullptr);
  EXPECT_EQ(searcher_raw->commit_calls(), 1);
  EXPECT_EQ(searcher_raw->committed_move(), move);
  EXPECT_EQ(match_task->state, TaskState::kNew);
  EXPECT_FALSE(match_task->coroutine.has_value());
  EXPECT_FALSE(match_task->response.has_value());
  ASSERT_EQ(match_task->move_uci_history.size(), 1u);
  EXPECT_EQ(match_task->move_uci_history[0], "e2e4");
  EXPECT_EQ(match_task->white_backend_slot, 0);
  EXPECT_EQ(match_task->black_backend_slot, 1);
  EXPECT_EQ(completed_future.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);

  ready_queue.close();
  runner.Stop();
}

TEST(MatchGameRunner, TerminalResultBuildsCompletedMatchGame) {
  ThreadSafeQueue<CompletedSearch> completion_queue;
  ThreadSafeQueue<std::unique_ptr<GameTask>> ready_queue;
  std::promise<CompletedMatchGame> completed_promise;
  auto completed_future = completed_promise.get_future();
  MatchGameRunner runner(MatchGameRunnerChannels{
      .completion_queue = &completion_queue,
      .ready_queue = &ready_queue,
      .on_completed_game =
          [&completed_promise](CompletedMatchGame completed_game) {
            completed_promise.set_value(std::move(completed_game));
          },
  });
  runner.Start();

  auto task = MakeMatchTask(std::make_unique<CommitRecordingSearcher>(),
                            TaskState::kNew,
                            /*white_backend_slot=*/1,
                            /*black_backend_slot=*/0);
  auto* match_task = dynamic_cast<MatchTask*>(task.get());
  ASSERT_NE(match_task, nullptr);
  match_task->move_uci_history.push_back("e2e4");
  match_task->move_uci_history.push_back("e7e5");

  SearchResult terminal;
  terminal.game_result = lczero::GameResult::WHITE_WON;
  ASSERT_TRUE(completion_queue.push(CompletedSearch{
      .task = std::move(task),
      .result = std::move(terminal),
  }));

  ASSERT_EQ(completed_future.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  CompletedMatchGame completed_game = completed_future.get();
  EXPECT_EQ(completed_game.game_result, lczero::GameResult::WHITE_WON);
  EXPECT_EQ(completed_game.white_backend_slot, 1);
  EXPECT_EQ(completed_game.black_backend_slot, 0);
  EXPECT_EQ(completed_game.move_uci_history,
            std::vector<std::string>({"e2e4", "e7e5"}));

  ready_queue.close();
  runner.Stop();
}

}  // namespace
}  // namespace engine

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
