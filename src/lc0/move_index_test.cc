/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "move_index.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "utils/bititer.h"

namespace lczero {
namespace {

constexpr int kPolicySize = 1858;
constexpr std::array<int, 8> kTransforms = {
    NoTransform,
    FlipTransform,
    MirrorTransform,
    FlipTransform | MirrorTransform,
    TransposeTransform,
    TransposeTransform | FlipTransform,
    TransposeTransform | MirrorTransform,
    TransposeTransform | FlipTransform | MirrorTransform,
};

TEST(MoveIndex, IndexMoveIndexRoundTripAllTransforms) {
  for (int transform : kTransforms) {
    for (int idx = 0; idx < kPolicySize; ++idx) {
      const Move move = MoveFromNNIndex(idx, transform);
      const uint16_t roundtrip = MoveToNNIndex(move, transform);
      EXPECT_EQ(roundtrip, idx) << "transform=" << transform << " idx=" << idx
                                << " move=" << move.ToString(true);
    }
  }
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
