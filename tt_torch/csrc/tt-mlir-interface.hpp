// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdlib>
#include <iostream>

#include "tt/runtime/runtime.h"

namespace tt::torch {
std::shared_ptr<void> *Compile(std::string_view code);
std::string stableHLOAutomaticParallelization(std::string_view code,
                                              std::vector<int64_t> mesh_shape,
                                              size_t len_activations = 0,
                                              size_t len_graph_constants = 0);
std::string compileStableHLOToTTIR(std::string_view code);
std::tuple<std::shared_ptr<void> *, std::string>
compileTTIRToTTNN(std::string_view code, std::string_view system_desc_path,
                  size_t len_activations = 0, size_t len_graph_constants = 0,
                  bool enable_consteval = true, bool enable_optimizer = false);
void create_system_desc(tt::runtime::Device device,
                        std::string_view descriptor_path);
} // namespace tt::torch
