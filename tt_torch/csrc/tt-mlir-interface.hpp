#pragma once
#include <cstdlib>
#include <iostream>

#include "tt/runtime/runtime.h"

namespace tt::torch {
  tt::runtime::Binary Compile(std::string_view code);
}  // namespace tt::torch