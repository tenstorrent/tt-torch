# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

set(TT_MLIR_VERSION "4d28d1522647c51da4bd026fdf3e9f9cdde07f32")

if (TOOLCHAIN STREQUAL "ON")
    cmake_minimum_required(VERSION 3.20)
    project(ttmlir-toolchain LANGUAGES CXX C)
    execute_process(
        COMMAND git clone --recursive https://github.com/tenstorrent/tt-mlir.git ${PROJECT_SOURCE_DIR}/tt-mlir/src/tt-mlir
        RESULT_VARIABLE result1
        OUTPUT_VARIABLE output1
        ERROR_VARIABLE error_output1
    )
    if (NOT ${result1} EQUAL 0)
        message(FATAL_ERROR "Failed to clone tt-mlir: ${error_output1}")
    endif()
    execute_process(
        COMMAND /bin/bash -c "cd ${PROJECT_SOURCE_DIR}/tt-mlir/src/tt-mlir && git checkout ${TT_MLIR_VERSION}"
    )
else()
    set(TT_RUNTIME_DEBUG OFF CACHE BOOL "Enable runtime debugging")
    message(STATUS "TT_RUNTIME_DEBUG is set to: ${TT_RUNTIME_DEBUG}")
    set(BUILD_TTRT ON CACHE BOOL "Build the ttrt target")
    # Set the build command based on the BUILD_TTRT variable
    if(BUILD_TTRT)
        message(STATUS "Building ttrt")
        set(BUILD_COMMAND "${CMAKE_COMMAND}" --build . -- ttrt)
    else()
        message(STATUS "Not building ttrt")
        set(BUILD_COMMAND "")
    endif()

    include(ExternalProject)
    set(TT_MLIR_LIBRARY_PATH ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/lib/SharedLib/libTTMLIR.so)
    set(TT_MLIR_SHLO_LIBRARY_PATH ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/lib/libTTMLIRStableHLOToTTIR.a)
    set(TT_MLIR_PIPELINES_LIBRARY_PATH ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/lib/libMLIRTTIRPipelines.a)
    ExternalProject_Add(
        tt-mlir
        PREFIX ${TTTORCH_SOURCE_DIR}/third_party/tt-mlir
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
          -DTT_RUNTIME_DEBUG=${TT_RUNTIME_DEBUG}
          -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
          -DTTMLIR_ENABLE_DEBUG_STRINGS=${TT_RUNTIME_DEBUG}
        BUILD_COMMAND ${BUILD_COMMAND}
        GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git
        GIT_TAG ${TT_MLIR_VERSION}
        GIT_PROGRESS ON
        BUILD_BYPRODUCTS ${TT_MLIR_LIBRARY_PATH} ${TT_MLIR_SHLO_LIBRARY_PATH} ${TT_MLIR_PIPELINES_LIBRARY_PATH}
    )

    add_library(TTMLIRShared SHARED IMPORTED GLOBAL)
    set_target_properties(TTMLIRShared PROPERTIES EXCLUDE_FROM_ALL TRUE IMPORTED_LOCATION ${TT_MLIR_LIBRARY_PATH})
    add_dependencies(TTMLIRShared tt-mlir)

    add_library(TTMLIRStableHLOToTTIR STATIC IMPORTED GLOBAL)
    set_target_properties(TTMLIRStableHLOToTTIR PROPERTIES EXCLUDE_FROM_ALL TRUE IMPORTED_LOCATION ${TT_MLIR_SHLO_LIBRARY_PATH})
    add_dependencies(TTMLIRStableHLOToTTIR tt-mlir)

    add_library(MLIRTTIRPipelines STATIC IMPORTED GLOBAL)
    set_target_properties(MLIRTTIRPipelines PROPERTIES EXCLUDE_FROM_ALL TRUE IMPORTED_LOCATION ${TT_MLIR_PIPELINES_LIBRARY_PATH})
    add_dependencies(MLIRTTIRPipelines tt-mlir)

    install(FILES ${TT_MLIR_LIBRARY_PATH} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

endif()
