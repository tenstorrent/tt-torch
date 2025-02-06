// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt-mlir-interface.hpp"
#include "tt/runtime/types.h"
#include <optional>
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
#include "tt/runtime/detail/debug.h"
#include "tt/runtime/runtime.h"
#endif
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

static tt::target::DataType torch_scalar_type_to_dt(torch::ScalarType st) {
  switch (st) {
  case torch::ScalarType::Byte:
    return tt::target::DataType::UInt8;
  case torch::ScalarType::Char:
    return tt::target::DataType::UInt8;
  case torch::ScalarType::Short:
    return tt::target::DataType::UInt16;
  case torch::ScalarType::Int:
    return tt::target::DataType::UInt32;
  case torch::ScalarType::Long:
    return tt::target::DataType::UInt32;
  case torch::ScalarType::Half:
    return tt::target::DataType::Float16;
  case torch::ScalarType::Float:
    return tt::target::DataType::Float32;
  // case torch::ScalarType::Double:
  // case torch::ScalarType::ComplexHalf:
  // case torch::ScalarType::ComplexFloat:
  // case torch::ScalarType::ComplexDouble:
  // case torch::ScalarType::Bool:
  case torch::ScalarType::BFloat16:
    return tt::target::DataType::BFloat16;
  case torch::ScalarType::Bool:
    // tt-metal does not support boolean data type; so bfloat16 data type is
    // used instead.
    return tt::target::DataType::BFloat16;
  default:
    break;
  }
  assert(false && "Unsupported scalar type");
}

static torch::ScalarType dt_to_torch_scalar_type(tt::target::DataType df) {
  switch (df) {
  case tt::target::DataType::UInt8:
    return torch::ScalarType::Byte;
  case tt::target::DataType::UInt16:
    return torch::ScalarType::Short;
  case tt::target::DataType::UInt32:
    return torch::ScalarType::Int;
  case tt::target::DataType::Float16:
    return torch::ScalarType::Half;
  case tt::target::DataType::Float32:
    return torch::ScalarType::Float;
  case tt::target::DataType::BFloat16:
    return torch::ScalarType::BFloat16;
  default:
    break;
  }
  assert(false && "Unsupported scalar type");
}

static tt::runtime::Tensor create_tensor(const torch::Tensor &tensor) {
  auto data = std::shared_ptr<void>(
      tensor.data_ptr(),
      [tensor](void *) {
        (void)tensor;
      } // Capture tensor by value to increase ref count and keep it alive
  );

  auto shape =
      std::vector<uint32_t>(tensor.sizes().begin(), tensor.sizes().end());
  if (shape.empty()) {
    shape.push_back(1);
  }

  auto stride =
      std::vector<uint32_t>(tensor.strides().begin(), tensor.strides().end());
  if (stride.empty()) {
    stride.push_back(1);
  }

  return tt::runtime::createTensor(
      data, shape, stride, tensor.element_size(),
      torch_scalar_type_to_dt(tensor.scalar_type()));
}

template <typename T>
std::vector<int64_t> as_vec_int64(std::vector<T> const &vec) {
  std::vector<int64_t> result;
  result.reserve(vec.size());
  for (auto const &v : vec) {
    result.push_back(v);
  }
  return result;
}

static torch::Tensor create_torch_tensor(const tt::runtime::Tensor &tensor,
                                         const tt::runtime::TensorDesc &desc) {
  tt::runtime::Tensor untilized_tensor =
      tt::runtime::toHost(tensor, /*untilize=*/true);
  const std::vector<std::int64_t> shape = as_vec_int64(desc.shape);
  const std::vector<std::int64_t> stride = as_vec_int64((desc.stride));

  const tt::target::DataType rt_datatype =
      tt::runtime::getTensorDataType(untilized_tensor);
  const torch::ScalarType dataType = dt_to_torch_scalar_type(rt_datatype);

  at::Tensor torch_tensor = at::empty_strided(shape, stride, dataType);
  tt::runtime::Tensor rt_tensor = create_tensor(torch_tensor);
  tt::runtime::memcpy(rt_tensor, untilized_tensor);

  return torch_tensor;
}

std::vector<at::Tensor> run(std::vector<at::Tensor> &inputs,
                            py::bytes byte_stream) {

  std::string data_str = byte_stream;
  auto binary_ptr = std::shared_ptr<void>(
      new char[data_str.size()],
      [](void *ptr) { delete[] static_cast<char *>(ptr); } // Custom deleter
  );
  // Copy data into the allocated memory
  std::memcpy(binary_ptr.get(), data_str.data(), data_str.size());
  tt::runtime::Binary binary = tt::runtime::Binary(binary_ptr);

  auto device = tt::runtime::openDevice({0});

  int program_idx = 0;
  auto input_descs = binary.getProgramInputs(program_idx);

  for (int idx = 0; idx < inputs.size(); idx++) {
    if (!inputs[idx].is_contiguous()) {
      std::cout << "WARNING: Input " << idx
                << " is not contiguous. Converting to contiguous in-place."
                << std::endl;
      inputs[idx].set_(inputs[idx].contiguous());
    }
  }

  std::vector<tt::runtime::Tensor> rt_inputs;
  for (auto const &input : inputs) {
    rt_inputs.emplace_back(create_tensor(input));
  }

  std::vector<tt::runtime::Tensor> rt_outputs =
      tt::runtime::submit(device, binary, program_idx, rt_inputs);

  std::vector<at::Tensor> outputs;
  outputs.reserve(rt_outputs.size());
  const auto output_descs = binary.getProgramOutputs(program_idx);

  for (size_t i = 0; i < rt_outputs.size(); ++i) {
    auto &rt_output = rt_outputs.at(i);
    const auto &output_desc = output_descs.at(i);
    outputs.emplace_back(create_torch_tensor(rt_output, output_desc));
    tt::runtime::deallocateTensor(rt_output, /*force=*/true);
  }

  tt::runtime::closeDevice(device);

  return outputs;
}

std::string compile_stable_hlo_to_ttir(std::string_view code) {
  auto ret = tt::torch::compileStableHLOToTTIR(code);
  return ret;
}

std::tuple<py::bytes, std::string>
compile_ttir_to_bytestream(std::string_view code) {
  auto [binary_ptr, ttnn] = tt::torch::compileTTIRToTTNN(code);
  auto size = ::flatbuffers::GetSizePrefixedBufferLength(
      static_cast<const uint8_t *>(binary_ptr->get()));
  tt::runtime::Binary binary = tt::runtime::Binary(*binary_ptr);

  std::string data_str(static_cast<const char *>(binary_ptr->get()), size);
  delete binary_ptr;

  return std::make_tuple(py::bytes(data_str), ttnn);
}

py::bytes compile_stablehlo_to_bytestream(std::string_view code) {
  auto binary = tt::torch::Compile(code);
  auto size = ::flatbuffers::GetSizePrefixedBufferLength(
      static_cast<const uint8_t *>(binary->get()));

  std::string data_str(static_cast<const char *>(binary->get()), size);
  delete binary;
  return py::bytes(data_str);
}

std::string bytestream_to_json(py::bytes byte_stream) {
  std::string data_str = byte_stream;
  auto binary_ptr = std::shared_ptr<void>(
      new char[data_str.size()],
      [](void *ptr) { delete[] static_cast<char *>(ptr); } // Custom deleter
  );
  // Copy data into the allocated memory
  std::memset(binary_ptr.get(), 0, data_str.size());
  std::memcpy(binary_ptr.get(), data_str.data(), data_str.size());
  tt::runtime::Binary binary = tt::runtime::Binary(binary_ptr);
  return binary.asJson();
}

PYBIND11_MODULE(tt_mlir, m) {
  m.doc() = "tt_mlir";
  py::class_<tt::runtime::Binary>(m, "Binary")
      .def("getProgramInputs", &tt::runtime::Binary::getProgramInputs)
      .def("getProgramOutputs", &tt::runtime::Binary::getProgramOutputs)
      .def("asJson", &tt::runtime::Binary::asJson);
  m.def("compile", &compile_stablehlo_to_bytestream,
        "A function that compiles stableHLO to a bytestream");
  m.def("compile_ttir_to_bytestream", &compile_ttir_to_bytestream,
        "A function that compiles TTIR to a bytestream");
  m.def("compile_stable_hlo_to_ttir", &compile_stable_hlo_to_ttir,
        "A function that compiles stableHLO to TTIR");
  m.def("run", &run, "Push inputs and run binary");
  m.def("get_current_system_desc", &tt::runtime::getCurrentSystemDesc,
        "Get the current system descriptor");
  m.def("get_num_available_devices", &tt::runtime::getNumAvailableDevices,
        "Get the number of available devices");
  m.def("bytestream_to_json", &bytestream_to_json,
        "Convert the bytestream to json");

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  py::class_<tt::runtime::CallbackContext>(m, "CallbackContext");
  py::class_<tt::runtime::OpContext>(m, "OpContext");
  py::class_<tt::runtime::TensorDesc>(m, "TensorDesc")
      .def_readonly("shape", &tt::runtime::TensorDesc::shape)
      .def_readonly("stride", &tt::runtime::TensorDesc::stride)
      .def_readonly("itemsize", &tt::runtime::TensorDesc::itemsize)
      .def_readonly("dataType", &tt::runtime::TensorDesc::dataType);
  m.def("get_op_output_tensor", &tt::runtime::getOpOutputTensor);
  m.def("get_op_debug_str", &tt::runtime::getOpDebugString,
        "Get the debug string of the op");
  m.def("get_op_loc_info", &tt::runtime::getOpLocInfo,
        "Get the location info of the op");
  py::class_<tt::runtime::debug::Hooks>(m, "DebugHooks")
      .def_static(
          "get_debug_hooks",
          [](py::function func) {
            return tt::runtime::debug::Hooks::get(
                [func](tt::runtime::Binary binary,
                       tt::runtime::CallbackContext programContext,
                       tt::runtime::OpContext opContext) {
                  func(binary, programContext, opContext);
                });
          },
          "Get the debug hooks")
      .def("__str__", [](const tt::runtime::debug::Hooks &hooks) {
        std::stringstream os;
        os << hooks;
        return os.str();
      });

  /**
   * Cleanup code to force a well ordered destruction w.r.t. the GIL
   */
  auto cleanup_callback = []() {
    tt::runtime::debug::Hooks::get().unregisterHooks();
  };
  m.add_object("_cleanup", py::capsule(cleanup_callback));
  m.def("unregister_hooks",
        []() { tt::runtime::debug::Hooks::get().unregisterHooks(); });
  m.def("is_runtime_debug_enabled", []() -> bool { return true; });
#else
  m.def("is_runtime_debug_enabled", []() -> bool { return false; });
#endif
}
