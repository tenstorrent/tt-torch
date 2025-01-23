# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import mlir
import tt_mlir
from mlir.ir import Context, Location, Module
import numpy as np
import multiprocessing as mp
import time
import sys
import tempfile
import re
import os
import mlir.dialects.stablehlo as stablehlo

from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    Op,
    OpCompilationStatus,
    calculate_atol,
    calculate_pcc,
)


def parse_module_from_str(module_str: str):
    module = None
    with Context() as ctx:
        stablehlo.register_dialect(ctx)
        module = Module.parse(module_str)
    return module


class StableHLOOp:
    def __init__(self, op_id, shlo, original_shlo):
        self.op_id = op_id
        self.original_shlo = original_shlo
        self.shlo = shlo
        self.binary = ""
        self.ttir = ""
        self.ttnn = ""
        self.pcc = None
        self.atol = None
        self.compilation_status = OpCompilationStatus.CONVERTED_TO_STABLE_HLO

    def add_ttir_graph(self, ttir):
        self.ttir = ttir

    def add_ttnn_graph(self, ttnn):
        self.ttnn = ttnn

    def __str__(self):
        return (
            f"StableHLOOp(op_id={self.op_id}, \nshlo=\n{self.shlo},"
            f"\noriginal_shlo=\n{self.original_shlo}, \nttir=\n{self.ttir}, \nttnn=\n{self.ttnn},"
            f"\npcc={self.pcc}, \natol={self.atol},"
            f"\ncompilation_status={self.compilation_status})"
        )


class StableHLOExecutor:
    def __init__(
        self, module_str, compiler_config=None, required_pcc=0.99, required_atol=1e-2
    ):
        self.module_str = module_str
        self.parsed_module = parse_module_from_str(module_str)
        self.binary = None
        self.compiler_config = compiler_config or CompilerConfig()
        self.required_pcc = required_pcc
        self.required_atol = required_atol
        self.sub_modules = []
        self.get_ops_in_module(self.parsed_module)

    def get_ops_in_module(self, module):
        for func_op in module.body.operations:
            for block in func_op.regions[0].blocks:
                for op in block.operations:
                    if op.name.startswith(("func.", "return")):
                        continue
                    if op.name in ["stablehlo.pad", "stablehlo.reduce_window"]:
                        continue

                    inputs = {
                        operand.get_name(): str(operand.type) for operand in op.operands
                    }
                    args_str = ", ".join(f"{key}: {typ}" for key, typ in inputs.items())
                    result_type = str(op.result.type)
                    result_name = str(op.result.get_name())

                    new_module_str = f"""module {{
    func.func @main({args_str}) -> {result_type} {{
        {str(op)}
        return {result_name} : {result_type}
    }}
}}"""
                    op_obj = StableHLOOp(result_name, new_module_str, op)
                    op_obj.pcc = self.required_pcc
                    op_obj.atol = self.required_atol
                    self.sub_modules.append(op_obj)

    def compile_process(self, asm):
        ttir = tt_mlir.compile_stable_hlo_to_ttir(asm)
        binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
        return ttir, binary, ttnn

    def compile_op(self, op):
        parsed = parse_module_from_str(op.shlo)
        asm = parsed.operation.get_asm()

        ttir, binary, ttnn = self.compile_process(asm)

        op.add_ttir_graph(ttir)
        op.add_ttnn_graph(ttnn)
        op.binary = binary
        op.compilation_status = OpCompilationStatus.CONVERTED_TO_TTNN
        return

    def compile_op_by_op(self):
        for op in self.sub_modules:
            try:
                self.compile_op(op)
            except Exception as e:
                print(f"Error in compiling op {op.op_id}: {e}")

    def print_graph(self):
        for op in self.sub_modules:
            print(f"Running: {op.op_id}\n\n\n")
            print(op)
            print("\n\n\n")

    def __call__(self, *inputs):
        if (
            self.compiler_config.compile_depth == CompileDepth.EXECUTE
            or self.compiler_config.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
        ):
            print("Execution not supported for StableHLO")
            exit(1)

        elif self.compiler_config.compile_depth == CompileDepth.COMPILE_OP_BY_OP:
            print("Ok")
            self.compile_op_by_op()
            self.print_graph()
            return

        else:
            raise ValueError("Unsupported compilation depth")
