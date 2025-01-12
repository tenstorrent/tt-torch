
list(APPEND CMAKE_PREFIX_PATH ${TTTORCH_SOURCE_DIR}/env/venv/lib/python3.11/site-packages/pybind11)
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)


add_library(TT_TORCH_MLIR "tt-mlir-interface.cpp")

add_dependencies(TT_TORCH_MLIR
    tt-mlir
    TTMLIRShared
)

set_target_properties(TT_TORCH_MLIR PROPERTIES COMPILE_FLAGS "-fno-rtti")
install (TARGETS TT_TORCH_MLIR LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

target_include_directories(TT_TORCH_MLIR PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/include
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/include/ttmlir/Target/Common
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/include
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/runtime/include
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/stablehlo/
    ${TTMLIR_TOOLCHAIN_DIR}/include
    ${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/include
)

set(STABLEHLO_LIBS
    StablehloBase
    StablehloReferenceIndex
    StablehloReferenceErrors
    StablehloReferenceElement
    StablehloReferenceAxes
    StablehloReferenceValue
    StablehloReferenceTypes
    StablehloReferenceToken
    StablehloReferenceTensor
    StablehloReferenceScope
    StablehloReferenceProcessGrid
    StablehloReferenceProcess
    StablehloReferenceOps
    StablehloPasses
    ChloOps
    VhloOps
    VhloTypes
    StablehloOps
    StablehloRegister
    StablehloReferenceToken
    StablehloLinalgTransforms
    StablehloReferenceValue
    StablehloReferenceScope
    StablehloPasses
    StablehloReferenceOps
    StablehloSerialization
    InterpreterOps
    StablehloPortableApi
    StablehloPasses
    StablehloAssemblyFormat
    StablehloOps
    StablehloReferenceElement
    StablehloReferenceOps
    StablehloReferenceTensor
    StablehloRegister
    StablehloBase
    StablehloPasses
    StablehloReferenceErrors
    StablehloReferenceProcess
    StablehloReferenceToken
    StablehloSerialization
    StablehloBroadcastUtils
    StablehloPortableApi
    StablehloReferenceIndex
    StablehloReferenceProcessGrid
    StablehloReferenceTypes
    StablehloTypeInference
    StablehloLinalgTransforms
    StablehloReferenceAxes
    StablehloReferenceNumPy
    StablehloReferenceScope
    StablehloReferenceValue
)

target_link_libraries(TT_TORCH_MLIR PUBLIC
    LLVM
    MLIR
    TTMLIRShared
    TTMLIRStatic
    TTMLIRTosaToTTIR
    MLIRTTIRPipelines
    TTMLIRStableHLOToTTIR
    ${STABLEHLO_LIBS}
)
target_link_directories(TT_TORCH_MLIR PUBLIC
    ${TTMLIR_TOOLCHAIN_DIR}/lib
    ${TTTORCH_SOURCE_DIR}/install/lib
    ${TTTORCH_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/lib
)

set(TORCH_INSTALL_PREFIX ${TTTORCH_SOURCE_DIR}/env/venv/lib/python3.11/site-packages/torch)
set(CMAKE_PREFIX_PATH ${TORCH_INSTALL_PREFIX})

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
set(TARGET_NAME tt_mlir)
pybind11_add_module(${TARGET_NAME} bindings.cpp)

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
