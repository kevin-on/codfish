#include "actor/mcts/match_coordinator.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace engine {
namespace {

std::string EscapePgnTag(std::string value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    if (c == '\\' || c == '"') escaped.push_back('\\');
    escaped.push_back(c);
  }
  return escaped;
}

std::string GameResultToPgn(lczero::GameResult result) {
  switch (result) {
    case lczero::GameResult::WHITE_WON:
      return "1-0";
    case lczero::GameResult::BLACK_WON:
      return "0-1";
    case lczero::GameResult::DRAW:
      return "1/2-1/2";
    case lczero::GameResult::UNDECIDED:
      break;
  }
  throw std::runtime_error("cannot serialize undecided match game");
}

std::string JoinMoves(const std::vector<std::string>& moves) {
  std::string joined;
  for (std::size_t i = 0; i < moves.size(); ++i) {
    if (i > 0) joined += ' ';
    joined += moves[i];
  }
  return joined;
}

}  // namespace

MatchCoordinator::MatchCoordinator(
    SearchCoordinatorConfig config,
    std::unique_ptr<GameTaskFactory> task_factory,
    std::array<std::shared_ptr<InferenceBackend>, 2> backends,
    const FeatureEncoder* encoder)
    : config_(config),
      task_factory_(std::move(task_factory)),
      backends_(std::move(backends)),
      encoder_(encoder),
      worker_runtime_(config_.num_workers,
                      WorkerChannels{
                          .ready_queue = &ready_queue_,
                          .request_queue = &request_queue_,
                          .completion_queue = &completion_queue_,
                      }),
      inference_runtime_(
          InferenceChannels{
              .request_queue = &request_queue_,
              .ready_queue = &ready_queue_,
          },
          std::vector<std::shared_ptr<InferenceBackend>>{
              backends_.begin(),
              backends_.end(),
          },
          encoder_, config_.inference),
      game_runner_(MatchGameRunnerChannels{
          .completion_queue = &completion_queue_,
          .ready_queue = &ready_queue_,
          .on_completed_game =
              [this](CompletedMatchGame completed_game) {
                OnCompletedGame(std::move(completed_game));
              },
      }) {
  assert(config_.num_workers > 0);
  assert(task_factory_ != nullptr);
  assert(backends_[0] != nullptr);
  assert(backends_[1] != nullptr);
  assert(encoder_ != nullptr);
}

MatchCoordinator::~MatchCoordinator() { StopRun(); }

void MatchCoordinator::RunMatch(const RunMatchOptions& options) {
  bool cleanup_needed = true;
  try {
    StartRun(options);
    WaitForCompletedGames(options.num_games);
    StopRun();
    WritePgn(options);
    cleanup_needed = false;
  } catch (...) {
    if (cleanup_needed) StopRun();
    throw;
  }
}

void MatchCoordinator::StartRun(const RunMatchOptions& options) {
  assert(options.num_games > 0);
  if (options.output_pgn_path.empty()) {
    throw std::invalid_argument("output_pgn_path must be provided");
  }
  if ((options.num_games % 2) != 0) {
    throw std::invalid_argument("num_games must be even for balanced colors");
  }

  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (started_ || stopped_) {
      throw std::logic_error(
          "MatchCoordinator::RunMatch() cannot be called more than once");
    }
    completed_games_.clear();
    completed_games_count_ = 0;
    started_ = true;
  }

  game_runner_.Start();
  inference_runtime_.Start();
  worker_runtime_.Start();
  SeedGameTasks(options.num_games);
}

void MatchCoordinator::StopRun() {
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (!started_ || stopped_) return;
    stopped_ = true;
  }

  worker_runtime_.Stop();
  inference_runtime_.Stop();
  game_runner_.Stop();
}

void MatchCoordinator::WaitForCompletedGames(int num_games) {
  std::unique_lock<std::mutex> lock(state_mu_);
  completion_cv_.wait(
      lock, [this, num_games] { return completed_games_count_ >= num_games; });
}

void MatchCoordinator::OnCompletedGame(CompletedMatchGame completed_game) {
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    completed_games_.push_back(std::move(completed_game));
    ++completed_games_count_;
  }
  completion_cv_.notify_all();
}

void MatchCoordinator::SeedGameTasks(int num_games) {
  for (int i = 0; i < num_games; ++i) {
    std::unique_ptr<GameTask> task = task_factory_->Create();
    assert(task != nullptr);
    assert(task->searcher != nullptr);
    assert(task->state == TaskState::kNew);
    const bool pushed = ready_queue_.push(std::move(task));
    if (!pushed) {
      throw std::runtime_error("ready queue rejected seeded match task");
    }
  }
}

void MatchCoordinator::WritePgn(const RunMatchOptions& options) const {
  const std::filesystem::path parent = options.output_pgn_path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream stream(options.output_pgn_path,
                       std::ios::out | std::ios::trunc | std::ios::binary);
  if (!stream.is_open()) {
    throw std::runtime_error("failed to open match PGN output path");
  }

  for (std::size_t i = 0; i < completed_games_.size(); ++i) {
    const CompletedMatchGame& game = completed_games_[i];
    const std::string result = GameResultToPgn(game.game_result);
    stream << "[Event \"Codfish Eval Match\"]\n";
    stream << "[Site \"?\"]\n";
    stream << "[Date \"????.??.??\"]\n";
    stream << "[Round \"" << (i + 1) << "\"]\n";
    stream << "[White \""
           << EscapePgnTag(options.player_names[game.white_backend_slot])
           << "\"]\n";
    stream << "[Black \""
           << EscapePgnTag(options.player_names[game.black_backend_slot])
           << "\"]\n";
    stream << "[Result \"" << result << "\"]\n";
    stream << "[Termination \"normal\"]\n";
    if (!game.move_uci_history.empty()) {
      stream << "[Moves \"" << EscapePgnTag(JoinMoves(game.move_uci_history))
             << "\"]\n";
    }
    stream << "\n" << result << "\n\n";
  }
}

}  // namespace engine
