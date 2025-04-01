# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re
import json
import numpy as np
from enum import Enum, IntEnum
from collections.abc import Iterable

from pathlib import Path
import os
import torch
import math
import sys
import shutil

from tt_mlir import (
    open_mesh_device,
    close_mesh_device,
    create_sub_mesh_device,
    release_sub_mesh_device,
    MeshDeviceOptions,
    is_runtime_debug_enabled,
)


"""
The CompileDepth's below represent the different stages of the compilation
pipeline.

tt-torch has two entrypoints it can compile from: PyTorch and ONNX. At the
beginning of the compile flow these entrypoints follow different paths, but
converge early in the compilation pipeline. The flow is as follows:

                                 PyTorch nn.Module
                                         |
     ONNX ModelProto               Torch FX Graph
            |                            |
      Torch ONNX IR                Torch FX IR  <-----  (first MLIR modules)
             \-----Torch Backend IR-----/
                           |
                       StableHLO
                           |
                      TTIR Dialect
                           |
                      TTNN Dialect
                           |
                 Flatbuffer Executable
"""


class CompileDepth(Enum):
    TORCH_FX = 1
    STABLEHLO = 2
    TTNN_IR = 3
    COMPILE_OP_BY_OP = 4
    EXECUTE_OP_BY_OP = 5
    EXECUTE = 6


class OpCompilationStatus(IntEnum):
    NOT_STARTED = 0
    CREATED_GRAPH = 1
    CONVERTED_TO_TORCH_IR = 2
    CONVERTED_TO_TORCH_BACKEND_IR = 3
    CONVERTED_TO_STABLE_HLO = 4
    CONVERTED_TO_TTIR = 5
    CONVERTED_TO_TTNN = 6
    EXECUTED = 7


class OpByOpBackend(Enum):
    TORCH = 1
    STABLEHLO = 2


class Tensor:
    def __init__(self, shape):
        constrained_shape = []
        for dim in shape:
            if isinstance(dim, int):
                constrained_shape.append(dim)
            else:
                constrained_shape.append(-1)
        self.shape = constrained_shape

        self.data_type = ""
        self.buffer_type = ""
        self.layout = ""
        self.grid_shape = []

    def to_dict(self):
        return {
            "shape": self.shape,
            "data_type": self.data_type,
            "buffer_type": self.buffer_type,
            "layout": self.layout,
            "grid_shape": self.grid_shape,
        }


class Op:
    def __init__(self, torch_name, input_shapes, model_name):
        self.framework_op_name = torch_name
        self.num_ops = 1
        self.model_name = model_name
        self.input_shapes = input_shapes
        self.input_tensors = []
        self.output_shapes = []
        self.output_tensors = []
        self.frontend = "tt-torch"

        self.torch_ir_graph = ""
        self.stable_hlo_graph = ""
        self.stable_hlo_ops = []
        self.ttir_graph = ""
        self.ttnn_graph = ""
        self.json = ""
        self.binary = ""
        self.runtime_stack_dump = ""
        self.compilation_status = OpCompilationStatus.NOT_STARTED
        self.parsed_stable_hlo_ops = False
        self.parsed_ttnn_ops = False
        self.pcc = None
        self.atol = None

    def parse_json(self):
        # Replace inf with strings until https://github.com/tenstorrent/tt-mlir/issues/2151 is fixed
        self.json = re.sub(r":\s*-inf\s*([,}])", r': "-inf"\1', self.json)
        self.json = re.sub(r":\s*inf\s*([,}])", r': "inf"\1', self.json)
        binary = json.loads(self.json)

        def tensor_from_tensor_desc(desc):
            tensor = Tensor(desc["shape"])
            if "memory_desc" in desc["layout"]:
                tensor.data_type = desc["layout"]["memory_desc"]["data_type"]
                try:
                    tensor.buffer_type = desc["layout"]["memory_desc"]["memory_space"]
                except KeyError:

                    if "memory_config" in desc["layout"]["memory_desc"]:
                        # If the tensor is on device, the descriptor will have a "memory_config" field
                        tensor.buffer_type = desc["layout"]["memory_desc"][
                            "memory_config"
                        ]["buffer_type"]
                    else:
                        # If the tensor is on host, the descriptor will have a "storage_type" field and no "memory_config" field
                        tensor.buffer_type = desc["layout"]["memory_desc"][
                            "storage_type"
                        ]

                try:
                    tensor.layout = desc["layout"]["memory_desc"]["memory_layout"]
                except KeyError:
                    if "memory_config" in desc["layout"]["memory_desc"]:
                        # If the tensor is on device, the descriptor will have a "memory_config" field
                        tensor.layout = desc["layout"]["memory_desc"]["memory_config"][
                            "tensor_memory_layout"
                        ]
                    else:
                        # If the tensor is on host, there will be no "memory_config" and thus no "tensor_memory_layout" field
                        tensor.layout = ""
            try:
                grid_shape = desc["layout"]["core_range_set"][0]["size"]
                tensor.grid_shape = [grid_shape["x"], grid_shape["y"]]
            except KeyError:
                pass
            return tensor

        for inp in binary["programs"][0]["inputs"]:
            self.input_tensors.append(tensor_from_tensor_desc(inp["desc"]))

        for out in binary["programs"][0]["outputs"]:
            self.output_tensors.append(tensor_from_tensor_desc(out["desc"]))

    def print_shapes(self, shapes):
        output = []
        for shape in shapes:
            output.append(f"{shape}")
        return output

    def to_dict(self):
        def scrub_nan_inf(value):
            if isinstance(value, float):
                if math.isnan(value):
                    ret = "NaN"
                elif math.isinf(value):
                    ret = "Inf"
                else:
                    ret = f"{value:.2f}"
            else:
                ret = ""
            return ret

        pcc = scrub_nan_inf(self.pcc)
        atol = scrub_nan_inf(self.atol)

        if len(self.input_tensors) == 0:
            self.input_tensors = [
                Tensor(shp) for shp in self.input_shapes if isinstance(shp, Iterable)
            ]

        if len(self.output_tensors) == 0:
            self.output_tensors = [
                Tensor(shp) for shp in self.output_shapes if isinstance(shp, Iterable)
            ]

        return {
            "framework_op_name": self.framework_op_name,
            "torch_name": self.framework_op_name,  # For backward compatibility
            "frontend": self.frontend,
            "model_name": self.model_name,
            "input_shapes": self.print_shapes(self.input_shapes),
            "input_tensors": [tensor.to_dict() for tensor in self.input_tensors],
            "output_shapes": self.print_shapes(self.output_shapes),
            "output_tensors": [tensor.to_dict() for tensor in self.output_tensors],
            "num_ops": self.num_ops,
            "compilation_status": self.compilation_status,
            "parsed_stable_hlo_ops": self.parsed_stable_hlo_ops,
            "torch_ir_graph": self.torch_ir_graph,
            "stable_hlo_graph": self.stable_hlo_graph,
            "stable_hlo_ops": self.stable_hlo_ops,
            "ttir_graph": self.ttir_graph,
            "ttnn_graph": self.ttnn_graph,
            "runtime_stack_dump": self.runtime_stack_dump,
            "pcc": pcc,
            "atol": atol,
            "compiled_json": self.json,
        }

    def unique_key(self):
        key = self.framework_op_name
        for shape in self.input_shapes:
            if isinstance(shape, torch.Size):
                key += f"_{print_shape(shape)}"
            else:
                key += f"_{shape}"
        return key

    def add_torch_ir_graph(self, torch_ir_graph: str):
        self.torch_ir_graph = torch_ir_graph

    def add_stable_hlo_graph(self, stable_hlo_graph: str):
        self.stable_hlo_graph = stable_hlo_graph
        self.converted_to_stable_hlo = True
        self.parsed_stable_hlo_ops = False

    def add_ttir_graph(self, ttir_graph: str):
        self.ttir_graph = ttir_graph

    def add_ttnn_graph(self, ttnn_graph: str):
        self.ttnn_graph = ttnn_graph


class CompilerConfig:
    def __init__(self):
        self.compile_depth = CompileDepth.EXECUTE
        self.profile_ops = True
        self.torch_mlir_module = None
        self.stablehlo_mlir_module = None
        self.unique_ops = {}
        self.stable_hlo_ops = []
        self.model_name = ""
        self.results_path = "results/models/"
        self.single_op_timeout = 30
        self.op_by_op_backend = OpByOpBackend.TORCH
        self.enable_consteval = False
        self._consteval_parameters = False
        self._enable_intermediate_verification = False
        self.dump_debug = False
        self.dump_info = False
        self.check_all_ops_execute = False
        self._verify_op_by_op = False
        self.typecast_inputs = True
        self.cache_preprocessed_constants = False
        self.inline_parameters = False
        self.record_property = None
        self.record_property = lambda *args, **kwargs: None  # Default to no-op

        self.apply_environment_overrides()
        self.post_init()

    @property
    def verify_op_by_op(self):
        return self._verify_op_by_op

    @verify_op_by_op.setter
    def verify_op_by_op(self, value):
        assert isinstance(
            value, bool
        ), "enable_intermediate_verification must be a boolean"
        if value and self.compile_depth != CompileDepth.EXECUTE_OP_BY_OP:
            print(
                "WARNING: Setting verify_op_by_op to True but compile_depth is not set to EXECUTE_OP_BY_OP. This CompilerConfig flag will have no effect."
            )
        self._verify_op_by_op = value

    @property
    def enable_intermediate_verification(self):
        return self._enable_intermediate_verification

    @enable_intermediate_verification.setter
    def enable_intermediate_verification(self, value):
        assert isinstance(
            value, bool
        ), "enable_intermediate_verification must be a boolean"

        if value and not is_runtime_debug_enabled():
            raise RuntimeError(
                "attempting to set enable_intermediate_verification to True but tt_mlir was not built with runtime debug enabled. Rebuild this project with -DTT_RUNTIME_DEBUG=ON if you wish to verify intermediate results."
            )

        self._enable_intermediate_verification = True

    @property
    def consteval_parameters(self):
        return self._consteval_parameters

    @consteval_parameters.setter
    def consteval_parameters(self, value):
        self._consteval_parameters = value
        self.post_init()

    def apply_environment_overrides(self):
        compile_depth = os.environ.get("TT_TORCH_COMPILE_DEPTH")
        if compile_depth:
            self.compile_depth = CompileDepth[compile_depth]
        verify_op_by_op = os.environ.get("TT_TORCH_VERIFY_OP_BY_OP")
        if verify_op_by_op and int(verify_op_by_op):
            self.verify_op_by_op = True
        check_all_ops_execute = os.environ.get("TT_TORCH_CHECK_ALL_OPS_EXECUTE")
        if check_all_ops_execute:
            self.check_all_ops_execute = True
        verify_intermediates = os.environ.get("TT_TORCH_VERIFY_INTERMEDIATES")
        if verify_intermediates and int(verify_intermediates):
            self.enable_intermediate_verification = True
        enable_consteval = os.environ.get("TT_TORCH_CONSTEVAL")
        if enable_consteval and int(enable_consteval):
            self.enable_consteval = True
        consteval_parameters = os.environ.get("TT_TORCH_CONSTEVAL_PARAMETERS")
        if consteval_parameters and int(consteval_parameters):
            self.consteval_parameters = True
        inline_parameters = os.environ.get("TT_TORCH_INLINE_PARAMETERS")
        if inline_parameters and int(inline_parameters):
            self.inline_parameters = True
        dump_intermediates = os.environ.get("TT_TORCH_IR_LOG_LEVEL")
        if dump_intermediates:
            self.dump_debug = dump_intermediates == "DEBUG"
            self.dump_info = self.dump_debug or dump_intermediates == "INFO"

    def post_init(self):
        if self.consteval_parameters:
            torch._dynamo.config.inline_inbuilt_nn_modules = False
        else:
            torch._dynamo.config.inline_inbuilt_nn_modules = True

    def reset_unique_ops(self):
        self.unique_ops = {}

    def save_unique_ops(self):
        unique_op_dict = {}
        pytest_test = os.environ.get("PYTEST_CURRENT_TEST")
        # 'PYTEST_CURRENT_TEST' is unavailable for the scripts executed/invoked
        # with python command; use 'sys.argv[0]' instead.
        if pytest_test is None:
            pytest_test = sys.argv[0]

        # Keep slashes, replace all non-alphanumeric characters with underscore.
        pytest_test = re.sub(r"[^A-Za-z0-9_/]", "_", pytest_test)

        for key, op in self.unique_ops.items():
            unique_op_dict[key] = op.to_dict()
        output_file = Path(f"{self.results_path}{pytest_test}_unique_ops.json")
        print(f"#####  Saving unique ops to {output_file}#####  ")
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, "w") as f:
            json.dump(unique_op_dict, f)

        total_ops = len(unique_op_dict)
        num_executed_ops = 0
        for op in unique_op_dict.values():
            if op["compilation_status"] == OpCompilationStatus.EXECUTED:
                num_executed_ops += 1

        print(f"{num_executed_ops}/{total_ops} ops executed")
        if self.check_all_ops_execute:
            assert num_executed_ops == total_ops
            print(f"Verified all ops ran in {self.model_name}")

    def set_compile_depth(self, compile_depth: CompileDepth):
        self.compile_depth = compile_depth

    def set_profile_ops(self, profile_ops: bool):
        self.profile_ops = profile_ops

    def set_torch_mlir_module(self, mlir_module):
        self.torch_mlir_module = mlir_module

    def set_stablehlo_mlir_module(self, mlir_module):
        self.stablehlo_mlir_module = mlir_module
        # self.stable_hlo_ops, _ = parse_shlo_mlir(mlir_module)

    def to_dict(self):
        return {
            "compile_depth": serialize_enum(self.compile_depth),
            "profile_ops": self.profile_ops,
            "torch_mlir_module": self.torch_mlir_module,
            "stablehlo_mlir_module": self.stablehlo_mlir_module,
            "unique_ops": self.unique_ops,
            "stable_hlo_ops": self.stable_hlo_ops,
            "model_name": self.model_name,
            "results_path": self.results_path,
            "single_op_timeout": self.single_op_timeout,
            "enable_consteval": self.enable_consteval,
            "_consteval_parameters": self._consteval_parameters,
            "_enable_intermediate_verification": self._enable_intermediate_verification,
            "_verify_op_by_op": self._verify_op_by_op,
        }


def extract_shape(shape_str):
    assert shape_str.startswith("tensor<")
    assert shape_str.endswith(">")
    shape_str = shape_str[len("tensor<") : -1]
    dims = shape_str.split("x")
    return [int(dim) for dim in dims[:-1]]


def split_top(string, splitter=",", openers="([{", closers=")]}", whitespace=" \n\t"):
    outlist = []
    outstring = []

    depth = 0

    for c in string:
        if c in openers:
            depth += 1
        elif c in closers:
            depth -= 1

            if depth < 0:
                raise SyntaxError()

        if not depth and c == splitter:
            outlist.append("".join(outstring))
            outstring = []
        else:
            if len(outstring):
                outstring.append(c)
            elif c not in whitespace:
                outstring.append(c)

    outlist.append("".join(outstring))

    return outlist


def print_shape(shape):
    return "x".join([str(dim) for dim in shape])


def are_brackets_balanced(string):
    # Count open and closed brackets
    counts = {
        "round": {"open": string.count("("), "closed": string.count(")")},
        "curly": {"open": string.count("{"), "closed": string.count("}")},
        "square": {"open": string.count("["), "closed": string.count("]")},
    }

    # Check if all counts are balanced
    return all(count["open"] == count["closed"] for count in counts.values())


def parse_shlo_mlir(mlir_code, verbose=False):
    ops = []
    unique_ops = {}
    opBegin = False
    opString = ""
    if mlir_code is None:
        return ops, unique_ops

    for index, line in enumerate(mlir_code.splitlines()):
        line = line.strip()
        if not line.startswith("%"):
            if not opBegin:
                continue
            else:
                opString += line
        else:
            opBegin = True
            opString += line

        if not opBegin or not are_brackets_balanced(opString):
            continue
        if verbose:
            print(opString)
        output = opString.split(" = ")[0].strip()
        op_name = opString.split(" = ")[1].split(" ")[0]
        if op_name.startswith('"'):
            op_name = op_name.split('"')[1]
        elif "(" in op_name:
            op_name = op_name.split("(")[0]
        if verbose:
            print(f"  op_name: {op_name}")
        # reduce is special cased
        args_and_attr = opString.split(op_name)[1]
        if op_name == "stablehlo.reduce":
            op_name += "_" + args_and_attr.split("applies")[1].strip().split(" ")[0]
            dim = args_and_attr.split("dimensions = ")[1].split(" ")[0]
            attr = {"dim": dim}
            args = [args_and_attr.split(")")[0].strip("(")]
        elif op_name == "stablehlo.reduce_window":
            op_name += (
                "_"
                + "stablehlo"
                + args_and_attr.split("stablehlo")[1].strip().split(" ")[0]
            )
            # TODO: Add attributes
            attr = {}
            args = []
        else:
            args_and_attr = opString.split(op_name)[1]
            args_and_attr = args_and_attr[: args_and_attr.rfind(":")]
            args_and_attr = split_top(args_and_attr)
            args = []
            attr = {}
            for arg in args_and_attr:
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    attr[key.strip()] = value.strip()
                else:
                    args.append(arg.strip())
            if verbose:
                print(f"  args: {args}")
                print(f"  attr: {attr}")
        io_shapes = opString[opString.rfind(":") + 1 :].strip()
        io_shapes = io_shapes.split(" -> ")
        if len(io_shapes) == 1:
            # input and output shapes are the same
            input_shapes = io_shapes[0].split(", ")
            output_shapes = io_shapes[0].split(", ")
        else:
            input_shapes, output_shapes = io_shapes
            output_shapes = output_shapes.split(", ")
            input_shapes = input_shapes.strip("(").strip(")")
            input_shapes = input_shapes.split(", ")

        input_shapes = [extract_shape(shape) for shape in input_shapes]
        output_shapes = [extract_shape(shape) for shape in output_shapes]
        if verbose:
            print(f"  input_shapes: {input_shapes}")
            print(f"  output_shape: {output_shapes}")
        op = (output, op_name, args, attr, input_shapes, output_shapes, opString)
        ops.append(op)

        if op_name not in unique_ops:
            unique_ops[op_name] = {}

        if len(input_shapes) == 0:
            key = ""
        else:
            key = print_shape(input_shapes[0])
        for shape in input_shapes[1:]:
            key += f"x{print_shape(shape)}"
        if key not in unique_ops[op_name]:
            unique_ops[op_name][key] = {}
            unique_ops[op_name][key]["ops"] = []
            unique_ops[op_name][key]["num_ops"] = 1
        else:
            unique_ops[op_name][key]["num_ops"] += 1
        unique_ops[op_name][key]["ops"].append(opString)
        opBegin = False
        opString = ""
    return ops, unique_ops


def prepare_tensors(ret, golden):
    # Convert boolean tensors to float; so ATOL can be calculated.
    if golden.dtype == torch.bool:
        golden = golden.to(torch.float)

    # TTNN does not support all the data types. So convert 'ret' tensor type to
    # match 'golden' tensor type.
    if golden.dtype != ret.dtype:
        ret = ret.to(golden.dtype)

    return ret, golden


def calculate_atol(tensor, golden_tensor):
    if torch.equal(golden_tensor, tensor):
        return 0.0

    tensor, golden_tensor = prepare_tensors(tensor, golden_tensor)

    # Handle NaN and Inf by verifying if NaN and Inf exists at same location in
    # both tensors.
    tensor_nan_mask = torch.isnan(tensor)
    golden_nan_mask = torch.isnan(golden_tensor)
    tensor_inf_mask = torch.isinf(tensor)
    golden_inf_mask = torch.isinf(golden_tensor)

    # Compare NaN values (NaN == NaN is considered True).
    if not torch.all(tensor_nan_mask == golden_nan_mask):
        return torch.nan

    # Compare Inf values (Inf == Inf is considered True).
    if not torch.all(tensor_inf_mask == golden_inf_mask):
        return torch.inf

    # Verify if respective Inf values in both tensors have same sign.
    tensor_sign = torch.sign(tensor)
    golden_sign = torch.sign(golden_tensor)
    sign_comparison = tensor_sign == golden_sign
    masked_sign_comparison = torch.where(
        tensor_inf_mask, sign_comparison, torch.tensor(True)
    )
    if not torch.all(masked_sign_comparison):
        return torch.inf

    # Replace NaN values with 0 to avoid having NaN as ATOL
    tensor[tensor_nan_mask] = 0
    golden_tensor[golden_nan_mask]

    # Replace Inf values with 0 to avoid having NaN as ATOL
    tensor[tensor_inf_mask] = 0
    golden_tensor[golden_inf_mask] = 0

    return torch.max(torch.abs(golden_tensor - tensor)).item()


def calculate_pcc(tensor, golden_tensor):
    if torch.equal(golden_tensor, tensor):
        return 1.0

    tensor, golden_tensor = prepare_tensors(tensor, golden_tensor)
    return float(
        np.min(
            np.ma.corrcoef(
                np.ma.masked_invalid(
                    torch.squeeze(tensor).detach().float().numpy()
                ).flatten(),
                np.ma.masked_invalid(
                    torch.squeeze(golden_tensor).detach().float().numpy()
                ).flatten(),
            )
        )
    )


def serialize_enum(enum_value):
    return f"{enum_value.__class__.__name__}.{enum_value.name}"


class FileManager:
    @staticmethod
    def create_file(file_path):
        try:
            if not FileManager.check_directory_exists(os.path.dirname(file_path)):
                FileManager.create_directory(os.path.dirname(file_path))
            with open(file_path, "w") as file:
                file.write("")
        except OSError as e:
            raise OSError(f"error creating file: {e}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def create_directory(directory_path, exist_ok=False):
        try:
            os.makedirs(directory_path, exist_ok=exist_ok)
        except FileExistsError as e:
            raise FileExistsError(f"directory '{directory_path}' already exists")
        except OSError as e:
            raise OSError(f"error creating directory: {e}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def remove_file(file_path):
        try:
            os.remove(file_path)
        except FileNotFoundError:
            print(f"directory '{file_path}' not found - cannot remove")
        except PermissionError:
            raise PermissionError(
                f"insufficient permissions to remove file '{file_path}'"
            )
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def remove_directory(directory_path):

        try:
            shutil.rmtree(directory_path)
        except FileNotFoundError:
            print(f"directory '{directory_path}' not found - cannot remove")
        except PermissionError:
            raise PermissionError(
                f"insufficient permissions to remove directory '{directory_path}'"
            )
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def copy_file(dest_file_path, src_file_path):
        try:
            shutil.copy2(src_file_path, dest_file_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"the source file does not exist: '{src_file_path}'"
            )
        except PermissionError as e:
            raise PermissionError(
                f"permission denied: '{src_file_path}' or '{dest_file_path}'"
            )
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def check_file_exists(file_path):
        exists = False
        try:
            if os.path.exists(file_path):
                exists = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return exists

    @staticmethod
    def check_directory_exists(directory_path):
        exists = False
        try:
            if os.path.isdir(directory_path):
                exists = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return exists
