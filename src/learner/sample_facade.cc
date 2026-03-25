#include "learner/sample_facade.h"
#include "learner/sample_facade_internal.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "chess/board.h"
#include "chess/error.h"
#include "chess/position.h"
#include "chess/types.h"

namespace engine::learner::internal {
namespace {

std::array<float, 3> ToWdlTarget(const lczero::Position& position,
                                 lczero::GameResult game_result) {
  if (game_result == lczero::GameResult::UNDECIDED) {
    throw std::runtime_error("sample facade requires decided game result");
  }

  const lczero::GameResult stm_result =
      position.IsBlackToMove() ? -game_result : game_result;
  switch (stm_result) {
    case lczero::GameResult::WHITE_WON:
      return {1.0f, 0.0f, 0.0f};
    case lczero::GameResult::DRAW:
      return {0.0f, 1.0f, 0.0f};
    case lczero::GameResult::BLACK_WON:
      return {0.0f, 0.0f, 1.0f};
    case lczero::GameResult::UNDECIDED:
      break;
  }

  throw std::runtime_error("unsupported game result");
}

lczero::Move ParseAndValidateLegalMove(
    const lczero::Position& position, std::span<const lczero::Move> legal_moves,
    std::string_view move_uci, std::string_view label) {
  const lczero::Move parsed = position.GetBoard().ParseMove(move_uci);
  const auto it = std::find(legal_moves.begin(), legal_moves.end(), parsed);
  if (it == legal_moves.end()) {
    throw std::runtime_error("stored " + std::string(label) +
                             " is not legal: " + std::string(move_uci));
  }
  return *it;
}

void FillPolicyTarget(const RawPly& ply, const lczero::Position& position,
                      std::span<const lczero::Move> legal_moves,
                      uint16_t selected_idx,
                      std::span<float, lczero::kPolicySize> policy_target) {
  std::array<bool, lczero::kPolicySize> seen_policy{};
  for (const RawPolicyEntry& entry : ply.policy) {
    const lczero::Move move = ParseAndValidateLegalMove(
        position, legal_moves, entry.move_uci, "policy move");
    const uint16_t idx = lczero::MoveToNNIndex(move, 0);
    if (seen_policy[idx]) {
      throw std::runtime_error("duplicate policy move index");
    }
    seen_policy[idx] = true;
    policy_target[idx] = entry.prob;
  }

  if (!seen_policy[selected_idx]) {
    throw std::runtime_error("selected move missing from stored policy");
  }
}

}  // namespace

std::vector<EncodedSampleDraft> BuildEncodedSampleDrafts(
    const RawGame& raw_game) {
  if (raw_game.game_result == lczero::GameResult::UNDECIDED) {
    throw std::runtime_error("sample facade requires decided game result");
  }
  if (raw_game.plies.empty()) {
    return {};
  }
  if (!raw_game.initial_fen.has_value()) {
    throw std::runtime_error("raw game with plies is missing initial_fen");
  }

  lczero::PositionHistory history;
  history.Reset(lczero::Position::FromFen(*raw_game.initial_fen));

  FeatureEncoder encoder;
  std::vector<EncodedSampleDraft> samples;
  samples.reserve(raw_game.plies.size());
  for (const RawPly& ply : raw_game.plies) {
    const lczero::Position& position = history.Last();
    const std::vector<lczero::Move> legal_moves =
        position.GetBoard().GenerateLegalMoves();
    if (legal_moves.empty()) {
      throw std::runtime_error("stored ply has no legal moves at root");
    }

    const lczero::Move selected_move = ParseAndValidateLegalMove(
        position, legal_moves, ply.selected_move_uci, "selected move");

    EncodedSampleDraft draft;
    encoder.EncodeOne(history, draft.input);
    draft.wdl_target = ToWdlTarget(position, raw_game.game_result);
    draft.selected_move_uci = ply.selected_move_uci;
    FillPolicyTarget(ply, position, legal_moves,
                     lczero::MoveToNNIndex(selected_move, 0),
                     draft.policy_target);

    samples.push_back(std::move(draft));
    history.Append(selected_move);
  }

  return samples;
}

}  // namespace engine::learner::internal

namespace engine::learner {

EncodedGameSamples EncodeRawGame(const RawGame& raw_game) {
  EncodedGameSamples samples;
  const std::vector<internal::EncodedSampleDraft> drafts =
      internal::BuildEncodedSampleDrafts(raw_game);
  samples.sample_count = static_cast<int>(drafts.size());
  samples.inputs.reserve(drafts.size() * kInputElements);
  samples.policy_targets.reserve(drafts.size() * lczero::kPolicySize);
  samples.wdl_targets.reserve(drafts.size() * 3);

  for (const internal::EncodedSampleDraft& draft : drafts) {
    samples.inputs.insert(samples.inputs.end(), draft.input.begin(),
                          draft.input.end());
    samples.policy_targets.insert(samples.policy_targets.end(),
                                  draft.policy_target.begin(),
                                  draft.policy_target.end());
    samples.wdl_targets.insert(samples.wdl_targets.end(),
                               draft.wdl_target.begin(),
                               draft.wdl_target.end());
  }

  return samples;
}

}  // namespace engine::learner
