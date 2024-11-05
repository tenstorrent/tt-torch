// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdlib>
#include <iostream>

#include "tt/runtime/runtime.h"

namespace tt::torch {
std::shared_ptr<void>* Compile(std::string_view code);
} // namespace tt::torch
