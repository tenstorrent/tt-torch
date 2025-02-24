# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

set(TT_MLIR_VERSION "a566e0e41f0b7bb5742785e5df8d4341a7d98698")

if (BUILD_TOOLCHAIN)
    cmake_minimum_required(VERSION 3.20)
    project(ttmlir-toolchain LANGUAGES CXX C)
    if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir/src/tt-mlir)
        execute_process(
            COMMAND git clone --recursive https://github.com/tenstorrent/tt-mlir.git ${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir/src/tt-mlir
        )
    endif()
    execute_process(
        COMMAND /bin/bash -c "git checkout ${TT_MLIR_VERSION}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir/src/tt-mlir
    )

    execute_process(
        COMMAND pwd
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir/src/tt-mlir
    )

    execute_process(
        COMMAND /bin/bash ../../../build_mlir_env.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir/src/tt-mlir
    )
else()
    set(TT_RUNTIME_DEBUG OFF CACHE BOOL "Enable runtime debugging")
    message(STATUS "TT_RUNTIME_DEBUG is set to: ${TT_RUNTIME_DEBUG}")

    include(ExternalProject)
    set(TTMLIR_PREFIX ${TTTORCH_SOURCE_DIR}/third_party/tt-mlir)
    set(TTMLIR_INSTALL_PREFIX ${TTMLIR_PREFIX}/install)
    set(TT_MLIR_COMPILER_LIBRARY_PATH ${TTMLIR_INSTALL_PREFIX}/lib/libTTMLIRCompiler.so)
    set(TT_MLIR_RUNTIME_LIBRARY_PATH ${TTMLIR_INSTALL_PREFIX}/lib/libTTMLIRRuntime.so)

    set(TTMLIR_BUILD_DIR ${TTMLIR_PREFIX}/src/tt-mlir-build)

    ExternalProject_Add(
        tt-mlir
        PREFIX ${TTMLIR_PREFIX}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install ${TTMLIR_BUILD_DIR} --component SharedLib
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS
          -DCMAKE_BUILD_TYPE=Release
          -DCMAKE_C_COMPILER=clang-17
          -DCMAKE_CXX_COMPILER=clang++-17
          -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
          -DTT_RUNTIME_ENABLE_TTNN=ON
          -DTT_RUNTIME_ENABLE_TTMETAL=OFF
          -DTTMLIR_ENABLE_STABLEHLO=ON
          -DTTMLIR_ENABLE_RUNTIME=ON
          -DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF
          -DTT_RUNTIME_DEBUG=${TT_RUNTIME_DEBUG}
          -DCMAKE_INSTALL_PREFIX=${TTMLIR_INSTALL_PREFIX}
          -DTTMLIR_ENABLE_DEBUG_STRINGS=${TT_RUNTIME_DEBUG}
        GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git
        GIT_TAG ${TT_MLIR_VERSION}
        GIT_PROGRESS ON
        BUILD_BYPRODUCTS
            ${TT_MLIR_COMPILER_LIBRARY_PATH}
            ${TT_MLIR_RUNTIME_LIBRARY_PATH}
    )

    add_library(TTMLIRCompiler SHARED IMPORTED GLOBAL)
    set_target_properties(TTMLIRCompiler PROPERTIES EXCLUDE_FROM_ALL TRUE IMPORTED_LOCATION ${TT_MLIR_COMPILER_LIBRARY_PATH})
    add_dependencies(TTMLIRCompiler tt-mlir)
    add_library(TTMLIRRuntime SHARED IMPORTED GLOBAL)
    set_target_properties(TTMLIRRuntime PROPERTIES EXCLUDE_FROM_ALL TRUE IMPORTED_LOCATION ${TT_MLIR_RUNTIME_LIBRARY_PATH})
    add_dependencies(TTMLIRRuntime tt-mlir)

    install(DIRECTORY ${TTMLIR_INSTALL_PREFIX}/ DESTINATION "${CMAKE_INSTALL_PREFIX}" USE_SOURCE_PERMISSIONS)

endif()
