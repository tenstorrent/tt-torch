# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re
import json
import numpy as np
from enum import Enum, IntEnum
from pathlib import Path
import os
import torch


class CompileDepth(Enum):
    TORCH_MLIR = 1
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


class Op:
    def __init__(self, torch_name, input_shapes):
        self.torch_name = torch_name
        self.num_ops = 1
        self.input_shapes = input_shapes
        self.output_shapes = []

        self.stable_hlo_graph = ""
        self.stable_hlo_ops = []
        self.ttir_graph = ""
        self.ttnn_graph = ""
        self.compilation_status = OpCompilationStatus.NOT_STARTED
        self.parsed_stable_hlo_ops = False
        self.parsed_ttnn_ops = False

    def print_shapes(self, shapes):
        output = []
        for shape in shapes:
            output.append(f"{shape}")
        return output

    def to_dict(self):
        return {
            "torch_name": self.torch_name,
            "input_shapes": self.print_shapes(self.input_shapes),
            "output_shapes": self.print_shapes(self.output_shapes),
            "num_ops": self.num_ops,
            "compilation_status": self.compilation_status,
            "parsed_stable_hlo_ops": self.parsed_stable_hlo_ops,
            "stable_hlo_graph": self.stable_hlo_graph,
            "stable_hlo_ops": self.stable_hlo_ops,
            "ttir_graph": self.ttir_graph,
            "ttnn_graph": self.ttnn_graph,
        }

    def unique_key(self):
        key = self.torch_name
        for shape in self.input_shapes:
            if isinstance(shape, torch.Size):
                key += f"_{print_shape(shape)}"
            else:
                key += f"_{shape}"
        return key

    def add_stable_hlo_graph(self, stable_hlo_graph: str):
        self.stable_hlo_graph = stable_hlo_graph
        self.converted_to_stable_hlo = True
        try:
            self.stable_hlo_ops, _ = parse_shlo_mlir(stable_hlo_graph)
            self.parsed_stable_hlo_ops = True
        except:
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
        self.single_op_timeout = 5

        self.apply_environment_overrides()

    def apply_environment_overrides(self):
        compile_depth = os.environ.get("TT_TORCH_COMPILE_DEPTH")
        if compile_depth:
            self.compile_depth = CompileDepth[compile_depth]

    def save_unique_ops(self):
        unique_op_dict = {}
        pytest_test = os.environ.get("PYTEST_CURRENT_TEST")
        pytest_test = pytest_test.replace("::", "_").replace(".", "_")
        pytest_test = pytest_test.replace("[", "_").replace("]", "_")
        for key, op in self.unique_ops.items():
            unique_op_dict[key] = op.to_dict()
        output_file = Path(f"{self.results_path}{pytest_test}_unique_ops.json")
        print(f"#####  Saving unique ops to {output_file}#####  ")
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, "w") as f:
            json.dump(unique_op_dict, f)

    def set_compile_depth(self, compile_depth: CompileDepth):
        self.compile_depth = compile_depth

    def set_profile_ops(self, profile_ops: bool):
        self.profile_ops = profile_ops

    def set_torch_mlir_module(self, mlir_module):
        self.torch_mlir_module = mlir_module

    def set_stablehlo_mlir_module(self, mlir_module):
        self.stablehlo_mlir_module = mlir_module
        self.stable_hlo_ops, _ = parse_shlo_mlir(mlir_module)


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
        # if output == "%21":
        #   breakpoint()
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


def calculate_atol(tensor, golden_tensor):
    return torch.max(torch.abs(golden_tensor - tensor)).item()


def calculate_pcc(tensor, golden_tensor):
    return np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(tensor).detach().numpy()).flatten(),
            np.ma.masked_invalid(
                torch.squeeze(golden_tensor).detach().numpy()
            ).flatten(),
        )
    )
