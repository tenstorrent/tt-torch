# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

set(TT_MLIR_VERSION "912a96b984cf4c86470f2d075da72f96d88af872")
set(TORCH_MLIR_VERSION "a265d283357f572c5437f9217ea85fa9770d374c")

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

    set(TT_RUNTIME_ENABLE_PERF_TRACE OFF CACHE BOOL "Enable runtime Tracy perf tracing")
    message(STATUS "TT_RUNTIME_ENABLE_PERF_TRACE is set to: ${TT_RUNTIME_ENABLE_PERF_TRACE}")

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
          -DTT_RUNTIME_ENABLE_PERF_TRACE=${TT_RUNTIME_ENABLE_PERF_TRACE}
          -DTT_RUNTIME_ENABLE_TTMETAL=OFF
          -DTTMLIR_ENABLE_STABLEHLO=ON
          -DTTMLIR_ENABLE_RUNTIME=ON
          -DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF
          -DTT_RUNTIME_DEBUG=${TT_RUNTIME_DEBUG}
          -DCMAKE_INSTALL_PREFIX=${TTMLIR_INSTALL_PREFIX}
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

    # torch-mlir build

    set(TORCH_MLIR_PREFIX ${TTTORCH_SOURCE_DIR}/third_party/torch-mlir)
    set(TORCH_MLIR_INSTALL_PREFIX ${TORCH_MLIR_PREFIX}/install)

    set(TORCH_MLIR_BUILD_DIR ${TORCH_MLIR_PREFIX}/src/torch-mlir-build)
    ExternalProject_Add(
        torch-mlir
        PREFIX ${TORCH_MLIR_PREFIX}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install ${TORCH_MLIR_BUILD_DIR}
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DPython3_FIND_VIRTUALENV=ONLY
            -DMLIR_DIR=${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/mlir/
            -DLLVM_DIR=${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/llvm/
            -DCMAKE_C_COMPILER=clang-17
            -DCMAKE_CXX_COMPILER=clang++-17
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
            -DTORCH_MLIR_ENABLE_STABLEHLO=ON
            -DMLIR_ENABLE_BINDINGS_PYTHON=ON
            -DLLVM_TARGETS_TO_BUILD=host
            -DLLVM_ENABLE_ZSTD=OFF
            -DTORCH_MLIR_ENABLE_LTC=ON
            -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
            -DCMAKE_C_VISIBILITY_PRESET=hidden
            -DCMAKE_CXX_VISIBILITY_PRESET=hidden
            -DCMAKE_INSTALL_PREFIX=${TORCH_MLIR_INSTALL_PREFIX}

        GIT_SUBMODULES "externals/stablehlo"
        GIT_PROGRESS ON
        GIT_REPOSITORY https://github.com/tenstorrent/llvm-torch-mlir.git
        GIT_TAG ${TORCH_MLIR_VERSION}
    )

    install(DIRECTORY ${TORCH_MLIR_INSTALL_PREFIX}/python_packages/torch_mlir/torch_mlir DESTINATION "${CMAKE_INSTALL_PREFIX}" USE_SOURCE_PERMISSIONS)

endif()
