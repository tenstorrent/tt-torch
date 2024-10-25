#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "tt-mlir-interface.hpp"


namespace py = pybind11;


tt::runtime::Binary compile(std::string_view code) {
    return tt::torch::Compile(code);
}


static tt::target::DataType torch_scalar_type_to_dt(torch::ScalarType st)
{
    switch (st)
    {
        case torch::ScalarType::Byte: return tt::target::DataType::UInt8;
        case torch::ScalarType::Char: return tt::target::DataType::UInt8;
        case torch::ScalarType::Short: return tt::target::DataType::UInt16;
        case torch::ScalarType::Int: return tt::target::DataType::UInt32;
        case torch::ScalarType::Long: return tt::target::DataType::UInt32;
        case torch::ScalarType::Half: return tt::target::DataType::Float16;
        case torch::ScalarType::Float: return tt::target::DataType::Float32;
        // case torch::ScalarType::Double:
        // case torch::ScalarType::ComplexHalf:
        // case torch::ScalarType::ComplexFloat:
        // case torch::ScalarType::ComplexDouble:
        // case torch::ScalarType::Bool:
        case torch::ScalarType::BFloat16: return tt::target::DataType::BFloat16;
        default: break;
    }
    assert(false && "Unsupported scalar type");

}

static torch::ScalarType dt_to_torch_scalar_type(tt::target::DataType df)
{
    switch (df)
    {
        case tt::target::DataType::UInt8: return torch::ScalarType::Byte;
        case tt::target::DataType::UInt16: return torch::ScalarType::Short;
        case tt::target::DataType::UInt32: return torch::ScalarType::Int;
        case tt::target::DataType::Float16: return torch::ScalarType::Half;
        case tt::target::DataType::Float32: return torch::ScalarType::Float;
        case tt::target::DataType::BFloat16: return torch::ScalarType::BFloat16;
        default: break;
    }
    assert(false && "Unsupported scalar type");

}


static tt::runtime::Tensor create_tensor(const torch::Tensor& tensor)
{
    auto data = std::shared_ptr<void>(
        tensor.data_ptr(),
        [tensor](void*) { (void)tensor; }  // Capture tensor by value to increase ref count and keep it alive
    );

    auto shape = std::vector<uint32_t>(tensor.sizes().begin(), tensor.sizes().end());
    auto stride = std::vector<uint32_t>(tensor.strides().begin(), tensor.strides().end());

    return tt::runtime::createTensor(
        data, shape, stride, tensor.element_size(), torch_scalar_type_to_dt(tensor.scalar_type()));
}

template <typename T>
std::vector<int64_t> as_vec_int64(std::vector<T> const& vec)
{
    std::vector<int64_t> result;
    result.reserve(vec.size());
    for (auto const& v : vec)
    {
        result.push_back(v);
    }
    return result;
}
std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs, tt::runtime::Binary binary) {
    auto [system_desc, chip_ids] = tt::runtime::getCurrentSystemDesc();
    int dev_0 = chip_ids[0];
    auto device = tt::runtime::openDevice({dev_0});

    int program_idx = 0;
    auto input_descs = binary.getProgramInputs(program_idx);

    std::vector<tt::runtime::Tensor> rt_inputs;
    for (auto const& input : inputs)
    {
        rt_inputs.emplace_back(create_tensor(input));
    }

    std::vector<at::Tensor> outputs;
    std::vector<tt::runtime::Tensor> rt_outputs;
    std::vector<tt::runtime::TensorDesc> output_descs = binary.getProgramOutputs(program_idx);
    outputs.reserve(output_descs.size());
    for (auto const& desc : output_descs)
    {
        std::vector<std::int64_t> shape = as_vec_int64(desc.shape);
        std::vector<std::int64_t> stride = as_vec_int64(desc.stride);

        at::Tensor output = at::empty_strided(shape, stride, dt_to_torch_scalar_type(desc.dataType));
        outputs.emplace_back(std::move(output));
        rt_outputs.emplace_back(create_tensor(outputs.back()));
    }

    tt::runtime::Event _ = tt::runtime::submit(device, binary, program_idx, rt_inputs, rt_outputs);

    return outputs;
}

PYBIND11_MODULE(tt_mlir, m) {
    m.doc() = "tt_mlir";
    py::class_<tt::runtime::Binary>(m, "Binary")
        .def("getProgramInputs", &tt::runtime::Binary::getProgramInputs)
        .def("getProgramOutputs", &tt::runtime::Binary::getProgramOutputs)
        .def("as_json", &tt::runtime::Binary::asJson);
    m.def("compile", &compile, "A function that compiles a stableHLO model to a flatbuffer");
    m.def("run", &run, "Push inputs and run binary");
    m.def("get_current_system_desc", &tt::runtime::getCurrentSystemDesc,
            "Get the current system descriptor");
    m.def("get_num_available_devices", &tt::runtime::getNumAvailableDevices,
            "Get the number of available devices");
}
