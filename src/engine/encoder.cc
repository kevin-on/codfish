#include "encoder.h"

#include <algorithm>
#include <cassert>

#include "chess/bitboard.h"

namespace engine {
namespace {
std::span<uint8_t> PositionPlanesAt(std::span<uint8_t> input, int history_idx) {
  assert(history_idx >= 0);
  const std::size_t offset =
      static_cast<std::size_t>(history_idx) * kPositionPlanes * kBoardArea;
  return input.subspan(offset, kPositionPlanes * kBoardArea);
}

std::span<uint8_t> GlobalPlanesAt(std::span<uint8_t> input) {
  const std::size_t offset =
      static_cast<std::size_t>(kHistoryLength) * kPositionPlanes * kBoardArea;
  return input.subspan(offset, kGlobalPlanes * kBoardArea);
}

std::span<uint8_t> PlaneAt(std::span<uint8_t> planes, int plane_idx) {
  return planes.subspan(static_cast<std::size_t>(plane_idx) * kBoardArea,
                        kBoardArea);
}

void EncodeBitBoard(lczero::BitBoard bitboard, std::span<uint8_t> plane) {
  assert(plane.size() == kBoardArea);
  for (const lczero::Square sq : bitboard) {
    plane[sq.as_idx()] = 1;
  }
}

void FillPlane(std::span<uint8_t> plane, uint8_t value) {
  std::fill(plane.begin(), plane.end(), value);
}

void EncodePosition(const lczero::Position& pos, std::span<uint8_t> out) {
  assert(out.size() == kPositionPlanes * kBoardArea);
  FillPlane(out, 0);

  const auto& board = pos.GetBoard();
  EncodeBitBoard(board.ours() & board.pawns(), PlaneAt(out, kOurPawnsPlane));
  EncodeBitBoard(board.ours() & board.knights(),
                 PlaneAt(out, kOurKnightsPlane));
  EncodeBitBoard(board.ours() & board.bishops(),
                 PlaneAt(out, kOurBishopsPlane));
  EncodeBitBoard(board.ours() & board.rooks(), PlaneAt(out, kOurRooksPlane));
  EncodeBitBoard(board.ours() & board.queens(), PlaneAt(out, kOurQueensPlane));
  EncodeBitBoard(board.ours() & board.kings(), PlaneAt(out, kOurKingsPlane));
  EncodeBitBoard(board.theirs() & board.pawns(),
                 PlaneAt(out, kTheirPawnsPlane));
  EncodeBitBoard(board.theirs() & board.knights(),
                 PlaneAt(out, kTheirKnightsPlane));
  EncodeBitBoard(board.theirs() & board.bishops(),
                 PlaneAt(out, kTheirBishopsPlane));
  EncodeBitBoard(board.theirs() & board.rooks(),
                 PlaneAt(out, kTheirRooksPlane));
  EncodeBitBoard(board.theirs() & board.queens(),
                 PlaneAt(out, kTheirQueensPlane));
  EncodeBitBoard(board.theirs() & board.kings(),
                 PlaneAt(out, kTheirKingsPlane));

  if (pos.GetRepetitions() >= 1) {
    FillPlane(PlaneAt(out, kRepetitionAtLeast1Plane), 1);
  }
  if (pos.GetRepetitions() >= 2) {
    FillPlane(PlaneAt(out, kRepetitionAtLeast2Plane), 1);
  }
}

void EncodeGlobal(const lczero::Position& pos, std::span<uint8_t> out) {
  assert(out.size() == kGlobalPlanes * kBoardArea);
  FillPlane(out, 0);

  const auto& board = pos.GetBoard();
  const auto& castlings = board.castlings();
  if (castlings.we_can_00()) {
    FillPlane(PlaneAt(out, kOurKingsideCastlingPlane), 1);
  }
  if (castlings.we_can_000()) {
    FillPlane(PlaneAt(out, kOurQueensideCastlingPlane), 1);
  }
  if (castlings.they_can_00()) {
    FillPlane(PlaneAt(out, kTheirKingsideCastlingPlane), 1);
  }
  if (castlings.they_can_000()) {
    FillPlane(PlaneAt(out, kTheirQueensideCastlingPlane), 1);
  }

  const auto ep_markers = board.en_passant();
  if (!ep_markers.empty()) {
    const auto ep_file = (*ep_markers.begin()).file();
    // The board is always oriented from the side to move, so the encoded
    // en passant target is always on rank 6.
    const auto ep_target = lczero::Square(ep_file, lczero::kRank6);
    EncodeBitBoard(lczero::BitBoard::FromSquare(ep_target),
                   PlaneAt(out, kEnPassantPlane));
  }

  FillPlane(PlaneAt(out, kRule50Plane),
            static_cast<uint8_t>(pos.GetRule50Ply()));
}
}  // namespace

void FeatureEncoder::EncodeOne(std::span<const lczero::Position> positions,
                               std::span<uint8_t> out) const {
  assert(!positions.empty());
  assert(out.size() == kInputElements);
  FillPlane(out, 0);

  const int last_idx = static_cast<int>(positions.size()) - 1;
  const int positions_to_encode =
      std::min(static_cast<int>(positions.size()), kHistoryLength);
  for (int history_idx = 0; history_idx < positions_to_encode; ++history_idx) {
    EncodePosition(positions[static_cast<std::size_t>(last_idx - history_idx)],
                   PositionPlanesAt(out, history_idx));
  }

  EncodeGlobal(positions.back(), GlobalPlanesAt(out));
}

void FeatureEncoder::EncodeOne(const lczero::PositionHistory& hist,
                               std::span<uint8_t> out) const {
  assert(hist.GetLength() > 0);
  EncodeOne(hist.GetPositions(), out);
}

void FeatureEncoder::EncodeBatch(
    std::span<const std::span<const lczero::Position>> batch,
    std::span<uint8_t> out) const {
  assert(out.size() == kInputElements * batch.size());

  for (std::size_t idx = 0; idx < batch.size(); ++idx) {
    EncodeOne(batch[idx], out.subspan(idx * kInputElements, kInputElements));
  }
}

void FeatureEncoder::EncodeBatch(std::span<const lczero::PositionHistory> batch,
                                 std::span<uint8_t> out) const {
  assert(out.size() == kInputElements * batch.size());

  for (std::size_t idx = 0; idx < batch.size(); ++idx) {
    EncodeOne(batch[idx], out.subspan(idx * kInputElements, kInputElements));
  }
}
}  // namespace engine
