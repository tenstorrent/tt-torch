
list(APPEND CMAKE_PREFIX_PATH ${TTTORCH_SOURCE_DIR}/env/venv/lib/python3.11/site-packages/pybind11)
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)


add_library(TT_TORCH_MLIR "tt-mlir-interface.cpp")

add_dependencies(TT_TORCH_MLIR
    tt-mlir
    TTMLIRCompiler
    TTMLIRRuntime
)

set_target_properties(TT_TORCH_MLIR PROPERTIES COMPILE_FLAGS "-fno-rtti")
install (TARGETS TT_TORCH_MLIR LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

target_include_directories(TT_TORCH_MLIR PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/include
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/include/ttmlir/Target/Common
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/include
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/runtime/include
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/stablehlo/
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/shardy/
    ${TTMLIR_TOOLCHAIN_DIR}/include
    ${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo
    ${TTMLIR_TOOLCHAIN_DIR}/src/shardy
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/include
)

target_link_libraries(TT_TORCH_MLIR PUBLIC
    TTMLIRCompiler
    TTMLIRRuntime
)
target_link_directories(TT_TORCH_MLIR PUBLIC
    ${TTMLIR_TOOLCHAIN_DIR}/lib
    ${CMAKE_INSTALL_PREFIX}/lib
)

set(TORCH_INSTALL_PREFIX ${TTTORCH_SOURCE_DIR}/env/venv/lib/python3.11/site-packages/torch)
set(CMAKE_PREFIX_PATH ${TORCH_INSTALL_PREFIX})

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
set(TARGET_NAME tt_mlir)

pybind11_add_module(${TARGET_NAME} bindings.cpp)
add_compile_definitions(TTMLIR_ENABLE_STABLEHLO=1)
if (${TT_RUNTIME_DEBUG} MATCHES "ON")
    add_compile_definitions(TT_RUNTIME_DEBUG=1)
endif()

add_dependencies(${TARGET_NAME}
    TT_TORCH_MLIR
    tt-mlir
)

target_include_directories(${TARGET_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/include
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/include/ttmlir/Target/Common
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/include
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/runtime/include
    ${TTMLIR_TOOLCHAIN_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(${TARGET_NAME} PUBLIC
    TT_TORCH_MLIR
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)

target_link_libraries(${TARGET_NAME} PUBLIC coverage_config)

target_link_directories(${TARGET_NAME} PRIVATE
    ${TORCH_INSTALL_PREFIX}/lib
)
set_target_properties(${TARGET_NAME}
    PROPERTIES
    PREFIX ""  # Disable "lib" prefix.
    LIBRARY_OUTPUT_NAME ${TARGET_NAME}
    INSTALL_RPATH "$ORIGIN"
)
install (TARGETS ${TARGET_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
