# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/localdev/aknezevic/tt-torch/third_party/tt-mlir/src/tt-mlir")
  file(MAKE_DIRECTORY "/localdev/aknezevic/tt-torch/third_party/tt-mlir/src/tt-mlir")
endif()
file(MAKE_DIRECTORY
  "/localdev/aknezevic/tt-torch/third_party/tt-mlir/src/tt-mlir-build"
  "/localdev/aknezevic/tt-torch/third_party/tt-mlir"
  "/localdev/aknezevic/tt-torch/third_party/tt-mlir/tmp"
  "/localdev/aknezevic/tt-torch/third_party/tt-mlir/src/tt-mlir-stamp"
  "/localdev/aknezevic/tt-torch/third_party/tt-mlir/src"
  "/localdev/aknezevic/tt-torch/third_party/tt-mlir/src/tt-mlir-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/localdev/aknezevic/tt-torch/third_party/tt-mlir/src/tt-mlir-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/localdev/aknezevic/tt-torch/third_party/tt-mlir/src/tt-mlir-stamp${cfgdir}") # cfgdir has leading slash
endif()
