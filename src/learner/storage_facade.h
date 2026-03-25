#pragma once

#include <filesystem>

#include "learner/raw_types.h"

namespace engine::learner {

RawChunkFile ReadRawChunkFile(const std::filesystem::path& path);

}  // namespace engine::learner
