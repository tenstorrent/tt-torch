# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tt_mlir
import multiprocessing as mp
import os
import re
import typing


class StableHLOParser:
    def __init__(self, shlo_text):
        self.shlo_text = shlo_text
        self.operations = self._parse_operations()
        self.arguments = self._parse_arguments()

    def _parse_arguments(self):
        """
        Parse the function arguments from the module.

        :return: Dictionary of argument names to their tensor shapes
        """
        func_def_match = re.search(
            r"func\.func @main\((.*?)\)", self.shlo_text, re.DOTALL
        )
        if not func_def_match:
            return {}

        arg_list = func_def_match.group(1)
        arg_pattern = re.compile(r"(%arg\d+):\s*(tensor<[^>]+>)")

        return {
            arg_match.group(1): arg_match.group(2)
            for arg_match in arg_pattern.finditer(arg_list)
        }

    def _parse_operations(self):
        """
        Parse the StableHLO text and extract operation details.

        :return: Dictionary where key is op_index and value is a dictionary containing operation details
        """
        # op_pattern = re.compile(r'(%\d+)\s*=\s*(stablehlo\.[^\n]+(?:\n[^%\n]+)*)', re.MULTILINE | re.DOTALL)
        op_pattern = re.compile(
            r'(%\d+)\s*=\s*(["]?stablehlo\.[^\n]+(?:\n[^%\n]+)*["]?)',
            re.MULTILINE | re.DOTALL,
        )
        operations = {}

        # Parse operations
        for match in op_pattern.finditer(self.shlo_text):
            op_index = match.group(1)
            op_string = match.group(2).strip()

            # Extract output shape (last part after ->)
            output_shape_match = re.search(r"->\s*([^\n:]+)", op_string)
            output_shape = (
                output_shape_match.group(1).strip() if output_shape_match else ""
            )
            if output_shape == "":
                output_shape_match = re.search(r"tensor<([^>]*)>", op_string)
                output_shape = output_shape_match.group() if output_shape_match else ""
            # Merge multiline operation string into a single line
            op_string = " ".join(op_string.split())

            operations[op_index] = {
                "op_string": op_string,
                "output_shape": output_shape,
            }

        return operations

    def get_argument_shape(self, arg_name):
        """
        Get the shape of a specific argument.

        :param arg_name: Name of the argument (e.g., '%arg0')
        :return: Tensor shape of the argument or None if not found
        """
        return self.arguments.get(arg_name)

    def get_function_shape(self, op_id):
        try:
            return self.operations[op_id]["output_shape"]
        except KeyError:
            return None


class StableHLOGenerator(StableHLOParser):
    def __init__(self, shlo_text):
        super().__init__(shlo_text)
        self.stablehlo_modules = self._generate_modules()

    def _generate_modules(self):
        """Generates stablehlo modules for each operation in self.operations

        :return: List of dictionaries containing op index, op string, and generated stablehlo module string
        """
        modules = {}
        for op_index in self.operations:
            module_string = self._build_module_string(self.operations[op_index])
            modules[op_index] = {
                "op_string": self.operations[op_index]["op_string"],
                "stablehlo_module": module_string,
            }
        return modules

    def extract_arguments(self, operation_string):
        """Extracts arguments from a StableHLO operation string.

        Args:
            operation_string: The StableHLO operation string.

        Returns:
            A list of arguments, or an empty list if no arguments are found.
        """

        pattern = (
            r"%\w+"  # Matches strings starting with '%' followed by word characters
        )
        matches = re.findall(pattern, operation_string)
        return matches

    def _build_module_string(self, op_data):
        args = self.extract_arguments(op_data["op_string"])
        header_args = ""
        new_op_string = op_data["op_string"]
        idx = 0

        for arg in args:
            arg_shape = self.get_argument_shape(arg) or self.get_function_shape(arg)
            if arg_shape is None:
                print(f"Could not find shape for op {op_data}")
                return ""
            new_arg = "%new_arg" + str(idx)
            new_op_string = re.sub(arg, new_arg, new_op_string)
            header_args += new_arg + ": " + arg_shape
            if arg != args[-1]:
                header_args += ", "
            idx += 1

        module_str = f"""
module attributes {{
    func.func @main({header_args}) -> {op_data['output_shape']} {{
        %1 = {new_op_string}
        return %1 : {op_data['output_shape']}
    }}
}}
"""

        return module_str


if __name__ == "__main__":
    shlo_path = "tests/models/mmdetection/stablehlo_output.txt"
    with open(shlo_path, "r") as f:
        shlo_code = f.read()

    generator = StableHLOGenerator(shlo_code)
    for module in generator.stablehlo_modules:
        print(generator.stablehlo_modules[module])
        # print(f"Op Index: {module['op_index']}")
        # print(f"Op String: {module['op_string']}")
        # print(f"StableHLO Module:\n{module['stablehlo_module']}\n")
        # try:
        #     tt_mlir.compile_stable_hlo_to_ttir(module['stablehlo_module'])
        # except Exception as e:
        #     print(f"Error compiling {module['op_string']} at {module['op_index']}: {e}")
