/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The Codfish Authors
*/

#pragma once

#include <stdexcept>

namespace lczero {

class ChessError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace lczero
