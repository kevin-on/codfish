#pragma once

#include <cstddef>
#include <span>

#include "chess/position.h"

namespace engine {
inline constexpr int kBoardSize = 8;
inline constexpr int kBoardArea = kBoardSize * kBoardSize;
inline constexpr int kHistoryLength = 8;

enum PositionPlane : int {
  kOurPawnsPlane = 0,
  kOurKnightsPlane,
  kOurBishopsPlane,
  kOurRooksPlane,
  kOurQueensPlane,
  kOurKingsPlane,
  kTheirPawnsPlane,
  kTheirKnightsPlane,
  kTheirBishopsPlane,
  kTheirRooksPlane,
  kTheirQueensPlane,
  kTheirKingsPlane,
  kRepetitionAtLeast1Plane,
  kRepetitionAtLeast2Plane,
  kPositionPlaneCount,
};

enum GlobalPlane : int {
  kOurKingsideCastlingPlane = 0,
  kOurQueensideCastlingPlane,
  kTheirKingsideCastlingPlane,
  kTheirQueensideCastlingPlane,
  kEnPassantPlane,
  kRule50Plane,
  kGlobalPlaneCount,
};

inline constexpr int kPositionPlanes = kPositionPlaneCount;
inline constexpr int kGlobalPlanes = kGlobalPlaneCount;
inline constexpr int kInputPlanes =
    kPositionPlanes * kHistoryLength + kGlobalPlanes;
inline constexpr std::size_t kInputElements = kInputPlanes * kBoardArea;

class FeatureEncoder {
 public:
  // positions must be ordered oldest -> current.
  void EncodeOne(std::span<const lczero::Position> positions,
                 std::span<uint8_t> out) const;

  void EncodeOne(const lczero::PositionHistory& hist,
                 std::span<uint8_t> out) const;

  void EncodeBatch(std::span<const std::span<const lczero::Position>> batch,
                   std::span<uint8_t> out) const;

  void EncodeBatch(std::span<const lczero::PositionHistory> batch,
                   std::span<uint8_t> out) const;
};
}  // namespace engine
