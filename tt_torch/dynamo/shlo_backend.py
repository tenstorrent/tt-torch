# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch_mlir
import tt_mlir
from mlir.ir import Context, Module
from torch_mlir._mlir_libs._mlir import ir
import mlir.dialects.stablehlo as stablehlo
import re
import os
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

#########################################################
# Helper functions
#########################################################


def generate_random_inputs_for_shlo(module_str):
    func_match = re.search(r"func\.func @\w+\((.*?)\)", module_str)
    if not func_match:
        raise ValueError("Could not find function signature in StableHLO module")

    args_str = func_match.group(1)

    # Match the function arguments
    args = re.findall(r"%arg\d+:\s+tensor<([^>]+)>", args_str)

    inputs = []
    for shape_str in args:
        # Check if shape_str contains dimensions (e.g., 1x784x or f32)
        shape_dtype_match = re.match(r"([\dx]+)x([^>]+)", shape_str)

        if shape_dtype_match:
            # If it matches the pattern of dimensions and data type (like 1x784xf32)
            shape_str, dtype_str = shape_dtype_match.groups()

            dims = [int(dim) for dim in shape_str.split("x")]

            if dtype_str == "f32":
                inputs.append(torch.randn(dims, dtype=torch.float32))
            elif dtype_str == "f16":
                inputs.append(torch.randn(dims, dtype=torch.float16))
            elif dtype_str == "bf16":
                inputs.append(torch.randn(dims, dtype=torch.bfloat16))
            elif dtype_str == "f64":
                inputs.append(torch.randn(dims, dtype=torch.float64))
            elif dtype_str.startswith("i") or dtype_str.startswith("si"):
                bit_width = int(
                    re.search(r"i(\d+)|si(\d+)", dtype_str).group(1)
                    or re.search(r"i(\d+)|si(\d+)", dtype_str).group(2)
                )
                max_val = min(
                    100,
                    2 ** (bit_width - 1) - 1
                    if dtype_str.startswith("si")
                    else 2**bit_width - 1,
                )
                inputs.append(
                    torch.randint(
                        -max_val if dtype_str.startswith("si") else 0, max_val, dims
                    )
                )
            elif dtype_str.startswith("ui"):
                bit_width = int(re.search(r"ui(\d+)", dtype_str).group(1))
                max_val = min(100, 2**bit_width - 1)
                inputs.append(torch.randint(0, max_val, dims))
            else:
                raise ValueError(f"Unsupported datatype: {dtype_str}")

        else:
            # If the shape_str is just a data type (e.g., "f32"), treat it as a scalar or unknown shape
            if shape_str == "f32":
                inputs.append(torch.randn((), dtype=torch.float32))
            elif shape_str == "f16":
                inputs.append(torch.randn((), dtype=torch.float16))
            elif shape_str == "bf16":
                inputs.append(torch.randn((), dtype=torch.bfloat16))
            elif shape_str == "f64":
                inputs.append(torch.randn((), dtype=torch.float64))
            elif shape_str.startswith("i") or shape_str.startswith("si"):
                bit_width = int(
                    re.search(r"i(\d+)|si(\d+)", shape_str).group(1)
                    or re.search(r"i(\d+)|si(\d+)", shape_str).group(2)
                )
                max_val = min(
                    100,
                    2 ** (bit_width - 1) - 1
                    if shape_str.startswith("si")
                    else 2**bit_width - 1,
                )
                inputs.append(
                    torch.randint(
                        -max_val if shape_str.startswith("si") else 0, max_val, ()
                    )
                )
            elif shape_str.startswith("ui"):
                bit_width = int(re.search(r"ui(\d+)", shape_str).group(1))
                max_val = min(100, 2**bit_width - 1)
                inputs.append(torch.randint(0, max_val, ()))
            else:
                raise ValueError(f"Unsupported dtype: {shape_str}")

    return tuple(inputs)


def parse_module_from_str(module_str):
    module = None
    with Context() as ctx:
        stablehlo.register_dialect(ctx)
        module = Module.parse(module_str)
    return module


def print_shape(shape):
    return "x".join(str(s) for s in shape)


#########################################################
# StableHlo Op Class which inherits from Op Class
#########################################################


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
            "runtime_stack_dump": self.runtime_stack_dump,
            "pcc": pcc,
            "atol": atol,
            "compiled_json": self.json,
        }


#########################################################
# StablehloExecutor covers op-by-op CompileDepth Options
#########################################################


class StablehloExecutor(OpByOpExecutor):
    def __init__(
        self,
        module: Union[str, "torch_mlir._mlir_libs._mlir.ir.Module", None] = None,
        compiler_config=None,
        required_pcc=0.99,
        required_atol=1e-2,
        device=None,
    ):
        super().__init__(
            compiler_config=compiler_config,
            required_pcc=required_pcc,
            required_atol=required_atol,
            device=device,
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
        elif isinstance(module, (ir.Module, ir.Operation)):
            self.parsed_module = module
        else:
            raise ValueError(f"Invalid module type: {type(module)}")

    def add_program(self, program: torch.export.ExportedProgram, graph_constants):
        assert (
            self.compiler_config.compile_depth == CompileDepth.COMPILE_OP_BY_OP
            or self.compiler_config.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
        ) and self.compiler_config.op_by_op_backend == OpByOpBackend.STABLEHLO, (
            "gm can only be added in op by op mode"
        )
        self.program = program
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

    def get_ops_in_module(self, module_or_op):
        if hasattr(module_or_op, "body"):
            # This is likely a Module object
            module_body = module_or_op.body
        elif hasattr(module_or_op, "operation") and hasattr(
            module_or_op.operation, "regions"
        ):
            # This is likely an Operation that represents a module
            # Get the first region's first block
            module_body = module_or_op.operation.regions[0].blocks[0]
        else:
            raise TypeError(
                "Input must be either a Module object or an Operation representing a module"
            )

        for func_op in module_body.operations:
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

    def shlo_op_by_op(self):
        calculated = None
        num_ops = len(self.sub_ops)
        for idx, op in enumerate(self.sub_ops):
            print(f"Compiling {idx}/{num_ops}: {op.op_name}")
            try:
                binary, op = self.compile_op(op, None, None)
            except Exception as e:
                binary = None
                print(f"Failed to compile {idx}/{num_nodes}: {node.target}: {e}")
            if (
                self.compiler_config.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
                and binary is not None
            ):
                try:
                    inputs = generate_random_inputs_for_shlo(op.stable_hlo_graph)
                    inputs = self.typecast_inputs(inputs)
                    calculated, runtime_stack_dump = self.run_op(binary, *inputs)
                    self.compiler_config.unique_ops[
                        op.unique_key
                    ].runtime_stack_dump = runtime_stack_dump
                    print(f"Ran: {idx}/{num_ops}: {op.op_name}")
                    if calculated is None:
                        raise ValueError("Failed to execute")
                    op.compilation_status = OpCompilationStatus.EXECUTED
                except Exception as e:
                    print(f"Failed to execute {idx}/{num_ops}: {op.op_name}: {e}")
        self.binary = binary
        self.compiler_config.save_unique_ops()
        if self.execute_process is not None:
            self.execute_process.terminate()
            self.execute_process = None
        if self.stderror_redirected:
            os.unlink(self.file_stderr.name)
            self.stderror_redirected = False

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

        if self.compiler_config.compile_depth in (
            CompileDepth.EXECUTE_OP_BY_OP,
            CompileDepth.COMPILE_OP_BY_OP,
        ):
            self.shlo_op_by_op()
        else:
            assert False, "Invalid compile depth"

        if self.program is not None:
            return self.program.graph_module(
                *(self.graph_constants + tuple(self.program.buffers()) + inputs)
            )

        if self.binary is not None:
            return tt_mlir.run_end_to_end(inputs, self.binary)
