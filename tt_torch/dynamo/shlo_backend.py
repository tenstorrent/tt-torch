# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_mlir
import torch
import torch_mlir
from mlir.ir import Context, Location, Module
import numpy as np
import faulthandler
import multiprocessing as mp
import time
import sys
import tempfile
import re
import os
import mlir.dialects.stablehlo as stablehlo
from typing import Union

from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    Op,
    OpByOpBackend,
    OpCompilationStatus,
    calculate_atol,
    calculate_pcc,
)

from tt_torch.dynamo.executor import OpByOpExecutor


def generate_random_inputs_for_shlo(module_str):
    tensor_shapes = re.findall(r"tensor<([\dx]+)xf32>", module_str)
    inputs = []
    for shape_str in tensor_shapes:
        shape = [int(dim) for dim in shape_str.split("x")]
        inputs.append(torch.randn(shape, dtype=torch.float32))
    return inputs


def parse_module_from_str(module_str):
    module = None
    with Context() as ctx:
        stablehlo.register_dialect(ctx)
        module = Module.parse(module_str)
    return module


def compile_process(receiver, sender, ttir_event, ttnn_event, json_event):
    obj = receiver.get()
    faulthandler.disable()
    asm = obj["asm"]
    ttir = tt_mlir.compile_stable_hlo_to_ttir(asm)
    sender.put({"ttir": ttir})
    ttir_event.wait()
    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
    sender.put({"binary": binary, "ttnn": ttnn})
    ttnn_event.wait()
    sender.put({"json": tt_mlir.bytestream_to_json(binary)})
    json_event.wait()
    sys.exit(0)


def print_shape(shape):
    return "x".join(str(s) for s in shape)


class StablehloOp(Op):
    def __init__(
        self,
        model_name,
        op_id,
        original_shlo,
    ):
        super().__init__("", [], model_name)
        self.op_id = op_id
        self.compilation_status = OpCompilationStatus.CREATED_GRAPH
        self.original_shlo = original_shlo
        self.op_name = self._extract_op_name()
        self.input_shapes = self._extract_input_shapes()
        self.unique_key = ""
        self.set_unique_key()

    def _extract_op_name(self):
        # Extract operation name from either stablehlo or arith
        match = re.search(r"(stablehlo|arith)\.[a-zA-Z_]+", self.original_shlo)
        return match.group(0) if match else "unknown_op"

    def _extract_input_shapes(self):
        # Extract shapes handling both f32 and i64 types
        shapes = []
        # Look for tensor patterns with various types
        shape_patterns = re.findall(r"tensor<([0-9x]+)x[fi][0-9]+>", self.original_shlo)
        for shape_str in shape_patterns:
            # Convert each dimension to int, handling both single and multi-dimensional cases
            shape = tuple(int(dim) for dim in shape_str.split("x"))
            shapes.append(shape)
        return shapes

    def set_unique_key(self):
        """Generate a unique key based on operation name and input shapes"""
        key = self.op_name
        for shape in self.input_shapes:
            key += f"_{print_shape(shape)}"
        self.unique_key = key

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

        return {
            "frontend": self.frontend,
            "model_name": self.model_name,
            "input_shapes": self.print_shapes(self.input_shapes),
            "input_tensors": [tensor.to_dict() for tensor in self.input_tensors],
            "output_shapes": self.print_shapes(self.output_shapes),
            "output_tensors": [tensor.to_dict() for tensor in self.output_tensors],
            "num_ops": self.num_ops,
            "compilation_status": self.compilation_status,
            # "parsed_stable_hlo_ops": self.parsed_stable_hlo_ops,
            # "torch_ir_graph": self.torch_ir_graph,
            "stable_hlo_graph": self.stable_hlo_graph,
            # "stable_hlo_ops": self.stable_hlo_ops,
            "ttir_graph": self.ttir_graph,
            "ttnn_graph": self.ttnn_graph,
            # "runtime_stack_dump": self.runtime_stack_dump,
            "pcc": pcc,
            "atol": atol,
            "compiled_json": self.json,
        }


class StablehloExecutor(OpByOpExecutor):
    def __init__(
        self,
        module: Union[str, "torch_mlir._mlir_libs._mlir.ir.Module", None] = None,
        compiler_config=None,
        required_pcc=0.99,
        required_atol=1e-2,
    ):
        super().__init__(
            compiler_config=compiler_config,
            required_pcc=required_pcc,
            required_atol=required_atol,
        )
        self.parsed_module = None
        if module is not None:
            self.set_module(module)
        self.sub_ops = []
        self.get_ops_in_module(self.parsed_module)
        self.gm = None
        self.graph_constants = None

    def set_module(
        self, module: Union[str, "torch_mlir._mlir_libs._mlir.ir.Module"]
    ) -> None:
        if isinstance(module, str):
            self.parsed_module = parse_module_from_str(module)
        elif isinstance(module, torch_mlir._mlir_libs._mlir.ir.Module):
            self.parsed_module = module
        else:
            raise ValueError(f"Invalid module type: {type(module)}")

    def add_gm(self, gm: torch.fx.GraphModule, graph_constants):
        assert (
            self.compiler_config.compile_depth == CompileDepth.COMPILE_OP_BY_OP
            and self.compiler_config.op_by_op_backend == OpByOpBackend.STABLEHLO
        ), "gm can only be added in COMPILE_OP_BY_OP mode"
        self.gm = gm
        self.graph_constants = (
            (graph_constants,)
            if isinstance(graph_constants, (int, float))
            else tuple(graph_constants)
        )

    def get_stable_hlo_graph(self, op, inputs, **kwargs):
        if op.unique_key not in self.compiler_config.unique_ops:
            self.compiler_config.unique_ops[op.unique_key] = op
        else:
            self.compiler_config.unique_ops[op.unique_key].num_ops += 1
            return None, None

        module = parse_module_from_str(op.stable_hlo_graph)
        asm = module.operation.get_asm()
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_STABLE_HLO
        op.add_stable_hlo_graph(asm)
        return module, op

    def get_ops_in_module(self, module):
        for func_op in module.body.operations:
            for block in func_op.regions[0].blocks:
                for op in block.operations:
                    if op.name.startswith(("func.", "return")):
                        continue

                    inputs = {
                        operand.get_name(): str(operand.type) for operand in op.operands
                    }
                    args_str = ", ".join(f"{key}: {typ}" for key, typ in inputs.items())

                    # Handle multiple results in the operation
                    result_names = [str(result.get_name()) for result in op.results]
                    result_types = [str(result.type) for result in op.results]

                    # Construct the function signature based on the number of results
                    if len(result_names) == 1:
                        result_str = f"{result_types[0]}"
                        return_stmt = f"return {result_names[0]} : {result_types[0]}"
                    else:
                        result_str = f"({', '.join(result_types)})"
                        return_stmt = f"return ({', '.join(result_names)}) : ({', '.join(result_types)})"
                    # Build the new module string
                    new_module_str = f"""module {{
        func.func @main({args_str}) -> {result_str} {{
            {str(op)}
            {return_stmt}
        }}
    }}"""

                    opObj = StablehloOp(
                        model_name=self.compiler_config.model_name,
                        op_id=", ".join(result_names),
                        original_shlo=str(op),
                    )
                    opObj.add_stable_hlo_graph(new_module_str)
                    self.sub_ops.append(opObj)

    def compile_shlo_op_by_op(self):
        num_ops = len(self.sub_ops)
        for idx, op in enumerate(self.sub_ops):
            print(f"Compiling {idx}/{num_ops}: {op.op_name}")
            binary, op = self.compile_op(op, None, None)
        self.set_binary(binary)
        self.compiler_config.save_unique_ops(mode="stablehlo")

    def print_op(self, op):
        print(op.op_id)
        print(op.stable_hlo_graph)
        print(op.ttir_graph)
        print(op.ttnn_graph)
        print(op.json)

    def print_ops(self):
        for op in self.sub_ops:
            self.print_op(op)

    def __call__(self, *inputs):
        inputs = self.typecast_inputs(inputs)
        if self.compiler_config.compile_depth == CompileDepth.COMPILE_OP_BY_OP:
            self.compile_shlo_op_by_op()
            if self.gm is not None:
                return self.run_gm_op_by_op(*(inputs + self.graph_constants))
            return  # return nothing
        else:
            assert False, "Invalid compile depth"
