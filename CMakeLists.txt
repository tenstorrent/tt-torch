# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#


cmake_minimum_required(VERSION 3.20)

if (NOT DEFINED ENV{TTTORCH_ENV_ACTIVATED})
  message(FATAL_ERROR "tt-torch environment not activated. Please run 'source env/activate'.")
endif()

project(TT_TORCH)


set(TTTORCH_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(TTMLIR_TOOLCHAIN_DIR $ENV{TTMLIR_TOOLCHAIN_DIR})

set(CMAKE_C_COMPILER clang-17)
set(DCMAKE_CXX_COMPILER clang++-17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_INSTALL_PREFIX ${TTTORCH_SOURCE_DIR}/install)

link_directories(${TTMLIR_TOOLCHAIN_DIR}/lib)

add_subdirectory(tt_torch)
add_subdirectory(third_party)
