// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// c++ standard library includes
#include <ATen/core/TensorBody.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <variant>

// other library includes
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/extension.h>
#include <vector>

// tt-mlir includes
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
#include "tt/runtime/debug.h"
#endif
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"

// tt-torch includes
#include "tt-mlir-interface.hpp"

namespace py = pybind11;

py::object TORCH_TENSOR_PYCLASS = py::module::import("torch").attr("Tensor");

using HostReturnType = std::variant<std::vector<at::Tensor>, py::object>;

static tt::target::DataType torch_scalar_type_to_dt(torch::ScalarType st) {
  switch (st) {
  case torch::ScalarType::Byte:
    return tt::target::DataType::UInt8;
  case torch::ScalarType::Char:
    return tt::target::DataType::UInt8;
  case torch::ScalarType::Short:
    return tt::target::DataType::UInt16;
  case torch::ScalarType::UInt32:
    return tt::target::DataType::UInt32;
  case torch::ScalarType::Int:
    return tt::target::DataType::Int32;
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
    return torch::ScalarType::UInt32;
  case tt::target::DataType::Int32:
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
  auto shape =
      std::vector<uint32_t>(tensor.sizes().begin(), tensor.sizes().end());
  if (shape.empty()) {
    shape.push_back(1);
  }

  assert(tensor.is_contiguous() && "Cannot create runtime tensor from "
                                   "non-contiguous torch tensor");

  // Torch tensors which are contiguous may not always have a stride
  // attribute which indicates that the tensor is contiguous. This occurs
  // when the left-most dimension is 1. In such cases, when the left-most
  // dimension is 1, the stride value for that dimension will never be used,
  // and so they do not bother to compute it even when calling .contiguous().
  //
  // Our runtime expects that this stride is accurate. So, we will require
  // that this torch tensor is contiguous and then calculate a fully-accurate
  // stride for it.
  std::vector<uint32_t> stride = tt::runtime::utils::calculateStride(shape);

  return tt::runtime::createBorrowedHostTensor(
      tensor.data_ptr(), shape, stride, tensor.element_size(),
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

static torch::Tensor create_torch_tensor(const tt::runtime::Tensor &tensor) {
  tt::runtime::Tensor untilized_tensor =
      tt::runtime::toHost(tensor, /*untilize=*/true)[0];

  const std::vector<std::int64_t> shape =
      as_vec_int64(tt::runtime::getTensorShape(untilized_tensor));
  const std::vector<std::int64_t> stride =
      as_vec_int64(tt::runtime::getTensorStride(untilized_tensor));

  const tt::target::DataType rt_datatype =
      tt::runtime::getTensorDataType(untilized_tensor);
  const torch::ScalarType dataType = dt_to_torch_scalar_type(rt_datatype);

  at::Tensor torch_tensor = at::empty_strided(shape, stride, dataType);
  tt::runtime::Tensor rt_tensor = create_tensor(torch_tensor);
  tt::runtime::memcpy(rt_tensor, untilized_tensor);

  return torch_tensor;
}

std::string stable_hlo_automatic_parallelization(
    std::string_view code, std::vector<int64_t> mesh_shape,
    size_t len_activations, size_t len_graph_constants) {
  auto ret = tt::torch::stableHLOAutomaticParallelization(
      code, mesh_shape, len_activations, len_graph_constants);
  return ret;
}

std::string compile_stable_hlo_to_ttir(std::string_view code) {
  auto ret = tt::torch::compileStableHLOToTTIR(code);
  return ret;
}

std::tuple<py::bytes, std::string>
compile_ttir_to_bytestream(std::string_view code,
                           std::string_view sys_desc_path,
                           size_t len_activations, size_t len_graph_constants,
                           bool enable_consteval = true) {
  auto [binary_ptr, ttnn] =
      tt::torch::compileTTIRToTTNN(code, sys_desc_path, len_activations,
                                   len_graph_constants, enable_consteval);
  auto size = ::flatbuffers::GetSizePrefixedBufferLength(
      static_cast<const uint8_t *>(binary_ptr->get()));
  tt::runtime::Binary binary = tt::runtime::Binary(*binary_ptr);

  std::string data_str(static_cast<const char *>(binary_ptr->get()), size);
  delete binary_ptr;

  return std::make_tuple(py::bytes(data_str), ttnn);
}

std::string bytestream_to_json(py::bytes byte_stream) {
  std::string data_str = byte_stream;
  auto binary_ptr = std::shared_ptr<void>(
      new char[data_str.size()],
      [](void *ptr) { delete[] static_cast<char *>(ptr); } // Custom deleter
  );
  // Copy data into the allocated memory
  std::memcpy(binary_ptr.get(), data_str.data(), data_str.size());
  tt::runtime::Binary binary = tt::runtime::Binary(binary_ptr);
  return binary.asJson();
}

tt::runtime::Binary create_binary_from_bytestream(py::bytes byte_stream) {
  std::string data_str = byte_stream;
  auto binary_ptr = std::shared_ptr<void>(new char[data_str.size()],
                                          std::default_delete<char[]>());
  // Copy data into the allocated memory
  std::memcpy(binary_ptr.get(), data_str.data(), data_str.size());
  tt::runtime::Binary binary = tt::runtime::Binary(binary_ptr);
  return binary;
}

std::vector<tt::runtime::Tensor>
preprocess_inputs(tt::runtime::Device device, std::vector<at::Tensor> &inputs,
                  tt::runtime::Binary binary, uint32_t program_idx,
                  size_t tensor_start_idx) {
  for (int idx = 0; idx < inputs.size(); idx++) {
    if (!inputs[idx].is_contiguous()) {
      std::cout << "WARNING: Input " << idx
                << " is not contiguous. Converting to contiguous in-place."
                << std::endl;
      inputs[idx].set_(inputs[idx].contiguous());
    }
  }

  std::vector<tt::runtime::Tensor> rt_inputs;
  rt_inputs.reserve(inputs.size());
  for (const auto &input : inputs) {
    rt_inputs.emplace_back(create_tensor(input));
  }

  std::vector<tt::runtime::Tensor> rt_inputs_with_layout;
  rt_inputs_with_layout.reserve(inputs.size());

  for (size_t i = 0; i < rt_inputs.size(); ++i) {
    tt::runtime::Tensor &t = rt_inputs[i];
    tt::runtime::Layout layout =
        tt::runtime::getLayout(binary, program_idx, i + tensor_start_idx);
    tt::runtime::Tensor tensor = tt::runtime::toLayout(t, device, layout);
    tt::runtime::setTensorRetain(tensor, /*retain=*/true);
    rt_inputs_with_layout.push_back(tensor);
  }

  return rt_inputs_with_layout;
}

std::vector<tt::runtime::Tensor>
run_async(tt::runtime::Device device, tt::runtime::Binary &binary,
          uint32_t program_idx, std::vector<tt::runtime::Tensor> &rt_inputs) {
  std::vector<tt::runtime::Tensor> rt_outputs =
      tt::runtime::submit(device, binary, program_idx, rt_inputs);

  for (auto &rt_output : rt_outputs) {
    auto it = std::find_if(rt_inputs.begin(), rt_inputs.end(),
                           [&rt_output](const tt::runtime::Tensor &input) {
                             return input.handle == rt_output.handle;
                           });
    if (it != rt_inputs.end()) {
      // Output tensor is the same as an existing input tensor.
      rt_output = tt::runtime::toHost(rt_output, /*untilize=*/true)[0];
      tt::runtime::TensorDesc desc = tt::runtime::getTensorDesc(rt_output);
      tt::runtime::Tensor copied_tensor =
          tt::runtime::createOwnedHostTensor(nullptr, desc);
      tt::runtime::memcpy(copied_tensor, rt_output);
      rt_output = copied_tensor;
    }
  }
  return rt_outputs;
}

at::Tensor to_host_single_rt_tensor(tt::runtime::Tensor &rt_output) {
  at::Tensor output = create_torch_tensor(rt_output);
  tt::runtime::deallocateTensor(rt_output, /*force=*/true);

  return output;
}

at::Tensor to_host_single_rt_tensor_non_deallocating(tt::runtime::Tensor &rt_output) {
  at::Tensor output = create_torch_tensor(rt_output);
  return output;
}

py::object to_host_single_object(py::object obj) {
  assert(py::isinstance<py::dict>(obj) &&
         "Non-tensor type must be castable to a dictionary");
  py::dict attrs = obj.cast<py::dict>();
  for (auto &item : attrs) {
    std::string key = py::str(item.first);
    py::handle value = item.second;
    py::iterable value_wrapper;
    if (py::isinstance<tt::runtime::Tensor>(value)) {
      value_wrapper = py::list(py::make_tuple(value));
    } else {
      assert(py::isinstance<py::iterable>(value) &&
             "Value within object must be iterable");
      value_wrapper = value.cast<py::iterable>();
    }
    py::list res = py::list();
    for (auto &v : value_wrapper) {
      if (py::isinstance<tt::runtime::Tensor>(v)) {
        tt::runtime::Tensor rt_tensor = v.cast<tt::runtime::Tensor>();
        at::Tensor host_tensor = to_host_single_rt_tensor(rt_tensor);
        res.append(host_tensor);
      } else {
        res.append(v);
      }
    }
    if (res.size() == 1) {
      attrs[key.c_str()] = res[0];
    } else {
      attrs[key.c_str()] = res;
    }
  }
  return attrs;
}

HostReturnType to_host(py::args args) {
  std::vector<at::Tensor> outputs;

  // Handle the special case where the input is a single non-tensor object.
  bool is_single_non_tensor_obj =
      py::len(args) == 1 && !(py::isinstance<tt::runtime::Tensor>(args[0]) ||
                              py::isinstance<at::Tensor>(args[0]) ||
                              py::isinstance<py::tuple>(args[0]));
  if (is_single_non_tensor_obj) {
    return to_host_single_object(args[0]);
  }
  for (auto &arg : args) {
    if (py::isinstance<py::tuple>(arg)) {
      for (auto &item : arg) {
        if (py::isinstance<tt::runtime::Tensor>(item)) {
          tt::runtime::Tensor rt_tensor = item.cast<tt::runtime::Tensor>();
          outputs.emplace_back(to_host_single_rt_tensor(rt_tensor));
        }
        // Hack to get around the fact that pybind11 does not
        // recognize the torch.Tensor pyclass as the same as
        // the at::Tensor C++ class when inside a py::tuple.
        else if (py::isinstance(item, TORCH_TENSOR_PYCLASS)) {
          outputs.emplace_back(item.cast<at::Tensor>());
        }
      }
    } else if (py::isinstance<tt::runtime::Tensor>(arg)) {
      tt::runtime::Tensor rt_tensor = arg.cast<tt::runtime::Tensor>();
      outputs.emplace_back(to_host_single_rt_tensor(rt_tensor));
    } else if (py::isinstance<at::Tensor>(arg)) {
      outputs.emplace_back(arg.cast<at::Tensor>());
    }
  }

  return outputs;
}


HostReturnType to_host_non_deallocating(py::args args) {
  std::vector<at::Tensor> outputs;

  // Handle the special case where the input is a single non-tensor object.
  bool is_single_non_tensor_obj =
      py::len(args) == 1 && !(py::isinstance<tt::runtime::Tensor>(args[0]) ||
                              py::isinstance<at::Tensor>(args[0]) ||
                              py::isinstance<py::tuple>(args[0]));
  if (is_single_non_tensor_obj) {
    return to_host_single_object(args[0]);
  }
  for (auto &arg : args) {
    if (py::isinstance<py::tuple>(arg)) {
      for (auto &item : arg) {
        if (py::isinstance<tt::runtime::Tensor>(item)) {
          tt::runtime::Tensor rt_tensor = item.cast<tt::runtime::Tensor>();
          outputs.emplace_back(to_host_single_rt_tensor_non_deallocating(rt_tensor));
        }
        // Hack to get around the fact that pybind11 does not
        // recognize the torch.Tensor pyclass as the same as
        // the at::Tensor C++ class when inside a py::tuple.
        else if (py::isinstance(item, TORCH_TENSOR_PYCLASS)) {
          outputs.emplace_back(item.cast<at::Tensor>());
        }
      }
    } else if (py::isinstance<tt::runtime::Tensor>(arg)) {
      tt::runtime::Tensor rt_tensor = arg.cast<tt::runtime::Tensor>();
      outputs.emplace_back(to_host_single_rt_tensor_non_deallocating(rt_tensor));
    } else if (py::isinstance<at::Tensor>(arg)) {
      outputs.emplace_back(arg.cast<at::Tensor>());
    }
  }

  return outputs;
}

std::vector<at::Tensor> run(tt::runtime::Device device,
                            tt::runtime::Binary &binary, uint32_t program_idx,
                            std::vector<tt::runtime::Tensor> &rt_inputs) {
  std::vector<tt::runtime::Tensor> rt_outputs =
      tt::runtime::submit(device, binary, program_idx, rt_inputs);

  // Create all torch tensors first before deallocating any runtime tensors
  // This handles cases where the same tensor is returned multiple times
  std::vector<at::Tensor> outputs;
  outputs.reserve(rt_outputs.size());

  for (size_t i = 0; i < rt_outputs.size(); ++i) {
    auto &rt_output = rt_outputs.at(i);
    outputs.emplace_back(create_torch_tensor(rt_output));
  }

  for (size_t i = 0; i < rt_outputs.size(); ++i) {
    auto &rt_output = rt_outputs.at(i);
    tt::runtime::deallocateTensor(rt_output, /*force=*/true);
  }

  return outputs;
}

std::vector<at::Tensor> run_end_to_end(std::vector<at::Tensor> &inputs,
                                       py::bytes byte_stream) {

  tt::runtime::Binary binary = create_binary_from_bytestream(byte_stream);

  tt::runtime::Device device = tt::runtime::openMeshDevice({1, 1});

  const int program_idx = 0;

  std::vector<tt::runtime::Tensor> rt_inputs =
      preprocess_inputs(device, inputs, binary, program_idx, 0);

  std::vector<at::Tensor> outputs = run(device, binary, program_idx, rt_inputs);

  tt::runtime::closeMeshDevice(device);

  return outputs;
}

torch::Tensor
get_op_output_torch_tensor(tt::runtime::OpContext opContextHandle,
                           tt::runtime::CallbackContext programContextHandle) {

  tt::runtime::Tensor tensor =
      tt::runtime::getOpOutputTensor(opContextHandle, programContextHandle);

  // Some ops in a decomposed tfx node may not have valid output tensors (eg.
  // deallocate) For these, return an empty tensor

  if (tensor.handle == nullptr) {
    std::cout << "Warning: getOpOutputTensor returned a null tensor."
              << std::endl;
    return torch::Tensor(); // Return an empty PyTorch tensor
  }

  return create_torch_tensor(tensor);
}

PYBIND11_MODULE(tt_mlir, m) {
  m.doc() = "tt_mlir";
  py::enum_<::tt::runtime::DispatchCoreType>(m, "DispatchCoreType")
      .value("WORKER", ::tt::runtime::DispatchCoreType::WORKER)
      .value("ETH", ::tt::runtime::DispatchCoreType::ETH);
  py::class_<tt::runtime::Binary>(m, "Binary")
      .def("getProgramInputs", &tt::runtime::Binary::getProgramInputs)
      .def("getProgramOutputs", &tt::runtime::Binary::getProgramOutputs)
      .def("asJson", &tt::runtime::Binary::asJson);
  py::class_<tt::runtime::MeshDeviceOptions>(m, "MeshDeviceOptions")
      .def(py::init<>())
      .def_readwrite("mesh_offset", &tt::runtime::MeshDeviceOptions::meshOffset)
      .def_readwrite("device_ids", &tt::runtime::MeshDeviceOptions::deviceIds)
      .def_readwrite("num_hw_cqs", &tt::runtime::MeshDeviceOptions::numHWCQs)
      .def_readwrite("enable_program_cache",
                     &tt::runtime::MeshDeviceOptions::enableProgramCache)
      .def_property(
          "l1_small_size",
          [](const tt::runtime::MeshDeviceOptions &o) {
            return o.l1SmallSize.has_value() ? py::cast(o.l1SmallSize.value())
                                             : py::none();
          },
          [](tt::runtime::MeshDeviceOptions &o, py::handle value) {
            o.l1SmallSize = py::none().is(value)
                                ? std::nullopt
                                : std::make_optional(value.cast<size_t>());
          })
      .def_property(
          "dispatch_core_type",
          [](const tt::runtime::MeshDeviceOptions &o) {
            return o.dispatchCoreType.has_value()
                       ? py::cast(o.dispatchCoreType.value())
                       : py::none();
          },
          [](tt::runtime::MeshDeviceOptions &o, py::handle value) {
            o.dispatchCoreType =
                py::none().is(value)
                    ? std::nullopt
                    : std::make_optional(
                          value.cast<tt::runtime::DispatchCoreType>());
          });
  py::class_<tt::runtime::Device>(m, "Device");
  py::class_<tt::runtime::Tensor>(m, "Tensor");
  m.def("compile_ttir_to_bytestream", &compile_ttir_to_bytestream,
        py::arg("ttir"), py::arg("system_desc_path"),
        py::arg("len_activations") = 0, py::arg("len_graph_constants") = 0,
        py::arg("enable_consteval") = true,
        "A function that compiles TTIR to a bytestream");
  m.def("stable_hlo_automatic_parallelization",
        &stable_hlo_automatic_parallelization,
        "Run shardy automatic data parallelization pass on stableHLO");
  m.def("compile_stable_hlo_to_ttir", &compile_stable_hlo_to_ttir,
        "A function that compiles stableHLO to TTIR");
  m.def("open_mesh_device", &tt::runtime::openMeshDevice, py::arg("mesh_shape"),
        py::arg("options"),
        "Open a mesh of devices for execution using the new API and create "
        "system description");
  m.def("close_mesh_device", &tt::runtime::closeMeshDevice,
        py::arg("parent_mesh"), "Close the mesh device using new API");
  m.def("create_sub_mesh_device", &tt::runtime::createSubMeshDevice,
        py::arg("parent_mesh"), py::arg("mesh_shape"),
        py::arg("mesh_offset") = py::none(),
        "Create a sub-mesh device using the new API");
  m.def("release_sub_mesh_device", &tt::runtime::releaseSubMeshDevice,
        py::arg("sub_mesh"), "Release the sub-mesh device using the new API");
  m.def("deallocate_tensor", &tt::runtime::deallocateTensor, py::arg("tensor"),
        py::arg("force") = false, "Deallocate the tensor");
  m.def("preprocess_inputs", &preprocess_inputs,
        "Preprocess inputs for execution");
  m.def("run", &run,
        "Run the binary on pre-defined device and pre-processed inputs, "
        "returning the final torch tensors on host");
  m.def("run_async", &run_async,
        "Run the binary on pre-defined device and pre-processed inputs, "
        "returning the runtime tensors on device");
  m.def(
      "to_host",
      [](py::args args) {
        auto result = to_host(args);
        if (std::holds_alternative<std::vector<at::Tensor>>(result)) {
          return py::cast(std::get<std::vector<at::Tensor>>(result));
        } else {
          return std::get<py::object>(result);
        }
      },
      "Moves runtime tensors to host, either returning a list of torch tensors "
      "or a modified object containing torch tensors");
  m.def("to_host_non_deallocating", [](py::args args) {
        auto result = to_host_non_deallocating(args);
        if (std::holds_alternative<std::vector<at::Tensor>>(result)) {
          return py::cast(std::get<std::vector<at::Tensor>>(result));
        } else {
          return std::get<py::object>(result);
        }
      },
        "Moves runtime tensors to host, either returning a list of torch tensors "
        "or a modified object containing torch tensors");
  m.def("run_end_to_end", &run_end_to_end,
        "Run binary end to end, isolating all steps such as device opening, "
        "input preprocessing, execution and device closing");
  m.def("get_num_available_devices", &tt::runtime::getNumAvailableDevices,
        "Get the number of available devices");
  m.def("bytestream_to_json", &bytestream_to_json,
        "Convert the bytestream to json");
  m.def("create_binary_from_bytestream", &create_binary_from_bytestream,
        "Create a binary from bytestream");
  m.def("create_system_desc", &tt::torch::create_system_desc, py::arg("device"),
        py::arg("descriptor_path"), "Create a system description");

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  py::class_<tt::runtime::CallbackContext>(m, "CallbackContext");
  py::class_<tt::runtime::OpContext>(m, "OpContext");
  py::class_<tt::runtime::TensorDesc>(m, "TensorDesc")
      .def_readonly("shape", &tt::runtime::TensorDesc::shape)
      .def_readonly("stride", &tt::runtime::TensorDesc::stride)
      .def_readonly("itemsize", &tt::runtime::TensorDesc::itemsize)
      .def_readonly("dataType", &tt::runtime::TensorDesc::dataType);
  m.def("get_op_output_tensor", &tt::runtime::getOpOutputTensor);
  m.def("get_op_output_tensor_desc", &tt::runtime::getTensorDesc);
  m.def("get_op_output_torch_tensor", &get_op_output_torch_tensor);
  m.def("get_op_debug_str", &tt::runtime::getOpDebugString,
        "Get the debug string of the op");
  m.def("get_op_loc_info", &tt::runtime::getOpLocInfo,
        "Get the location info of the op");
  py::class_<tt::runtime::debug::Hooks>(m, "DebugHooks")
      .def_static(
          "get_debug_hooks",
          [](py::function func) {
            return tt::runtime::debug::Hooks::get(
                std::nullopt,
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
