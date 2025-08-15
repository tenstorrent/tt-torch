# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

# underlying tt-mlir version we will have tt-xla use
# commit cd51e36ab685499448552393c52caee506465c45 (HEAD -> jzx/xla_fusion, origin/jzx/xla_fusion)
set(TT_MLIR_VERSION "cd51e36ab685499448552393c52caee506465c45")

# tt-xla version to use
set(TTXLA_VERSION "643f9847bb12da60b6abd9561c275dd0cc499724") # jzx/persistent

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

    set(TTMLIR_ENABLE_OPMODEL OFF CACHE BOOL "Enable op model for optimized compile")
    message(STATUS "TTMLIR_ENABLE_OPMODEL is set to: ${TTMLIR_ENABLE_OPMODEL}")

    include(ExternalProject)
    set(TTMLIR_BUILD_DIR ${TTMLIR_PREFIX}/src/tt-mlir-build)

    set(TTXLA_PREFIX ${TTTORCH_SOURCE_DIR}/third_party/tt-xla)
    set(TTXLA_INSTALL_PREFIX ${TTXLA_PREFIX}/src/tt-xla/install)
    set(TTXLA_PJRT_DYLIB ${TTXLA_INSTALL_PREFIX}/lib/pjrt_plugin_tt.so)
    set(TTNN_DYLIB ${TTXLA_INSTALL_PREFIX}/lib/_ttnncpp.so)
    set(TTXLA_LIBDEVICE_DYLIB ${TTXLA_INSTALL_PREFIX}/lib/libdevice.so)
    set(TTMETAL_DYLIB ${TTXLA_INSTALL_PREFIX}/lib/libtt_metal.so)
    set(TTXLA_BUILD_DIR ${TTXLA_PREFIX}/src/tt-xla-build)

    set(TT_MLIR_COMPILER_LIBRARY_PATH ${TTXLA_INSTALL_PREFIX}/lib/libTTMLIRCompiler.so)
    set(TT_MLIR_RUNTIME_LIBRARY_PATH ${TTXLA_INSTALL_PREFIX}/lib/libTTMLIRRuntime.so)

    ExternalProject_Add(
        tt-xla
        PREFIX ${TTXLA_PREFIX}
        INSTALL_COMMAND
            ${CMAKE_COMMAND} --install ${TTXLA_BUILD_DIR} --prefix ${TTXLA_INSTALL_PREFIX} &&
            chmod -R +x ${TTXLA_INSTALL_PREFIX}
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Debug
            -DCMAKE_C_COMPILER=clang-17
            -DCMAKE_CXX_COMPILER=clang++-17
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
            -DUSE_CUSTOM_TT_MLIR_VERSION=ON
            -DTT_MLIR_VERSION=${TT_MLIR_VERSION}
            -DTT_RUNTIME_DEBUG=${TT_RUNTIME_DEBUG}
            -DCMAKE_INSTALL_PREFIX=${TTXLA_INSTALL_PREFIX}
            -DTTMLIR_ENABLE_PERF_TRACE=${TT_RUNTIME_ENABLE_PERF_TRACE}
        GIT_REPOSITORY https://github.com/tenstorrent/tt-xla.git
        GIT_TAG ${TTXLA_VERSION}
        GIT_PROGRESS ON
        BUILD_BYPRODUCTS
            ${TTXLA_PJRT_DYLIB}
            ${TTNN_DYLIB}
            ${TTXLA_LIBDEVICE_DYLIB}
            ${TTMETAL_DYLIB}
            ${TT_MLIR_COMPILER_LIBRARY_PATH}
            ${TT_MLIR_RUNTIME_LIBRARY_PATH}
    )

    add_library(TTMLIRCompiler SHARED IMPORTED GLOBAL)
    set_target_properties(TTMLIRCompiler PROPERTIES EXCLUDE_FROM_ALL TRUE IMPORTED_LOCATION ${TT_MLIR_COMPILER_LIBRARY_PATH})
    add_dependencies(TTMLIRCompiler tt-xla)
    add_library(TTMLIRRuntime SHARED IMPORTED GLOBAL)
    set_target_properties(TTMLIRRuntime PROPERTIES EXCLUDE_FROM_ALL TRUE IMPORTED_LOCATION ${TT_MLIR_RUNTIME_LIBRARY_PATH})
    add_dependencies(TTMLIRRuntime tt-xla)
    install(DIRECTORY ${TTXLA_INSTALL_PREFIX}/ DESTINATION "${CMAKE_INSTALL_PREFIX}" USE_SOURCE_PERMISSIONS)
    set(TTMLIR_PREFIX ${TTXLA_PREFIX}/src/tt-xla/third_party/tt-mlir/)
    set(TTMETAL_PREFIX ${TTMLIR_PREFIX}/src/tt-mlir/third_party/tt-metal/)
    install(DIRECTORY ${TTMETAL_PREFIX}/src/tt-metal/build/ DESTINATION "${CMAKE_INSTALL_PREFIX}/tt-metal" USE_SOURCE_PERMISSIONS)

    # torch-mlir build
    # torch-mlir is using branch tt_torch/main
    # see issue https://github.com/tenstorrent/tt-torch/issues/671
    set(TORCH_MLIR_VERSION "be21b75d7fc2a45d5cccf5ac5d8d9448106735bd")
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

    # tt-forge-models - Python-only project, no need to build.
    set(TT_FORGE_MODELS_VERSION "8e05e115ffd1c403f22499abf43adc02e9a73cd2")
    ExternalProject_Add(
        tt_forge_models
        SOURCE_DIR ${TTTORCH_SOURCE_DIR}/third_party/tt_forge_models
        GIT_REPOSITORY https://github.com/tenstorrent/tt-forge-models.git
        GIT_TAG ${TT_FORGE_MODELS_VERSION}
        GIT_PROGRESS ON
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
endif()
