#pragma once

#include <cstdint>

#include "chess/types.h"

namespace lczero {

uint16_t MoveToNNIndex(Move move, int transform);
Move MoveFromNNIndex(int idx, int transform);

}  // namespace lczero
