# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import mlir
import mlir.ir as ir
from mlir.ir import Context, Location, Module
import mlir.dialects.stablehlo as stablehlo
from typing import List, Dict, Any


def get_module_from_str(module_str: str):
    module = None
    with ir.Context() as ctx:
        stablehlo.register_dialect(ctx)
        module = Module.parse(module_str)
    return module


def get_ops_in_module(module: mlir.ir.Module) -> List[str]:
    """
    Get all operations in a module, excluding func.func and return ops.

    Args:
        module: MLIR module
    Returns:
        List of operation names
    """
    ops = []
    # Iterate through all functions in the module
    for func_op in module.body.operations:
        # Iterate through all blocks in the function
        for block in func_op.regions[0].blocks:
            # Iterate through all operations in the block
            for op in block.operations:
                # Skip return operations
                input_types = []
                result_type = None
                if not op.name.startswith(("func.", "return")):
                    ops.append(op.name)
                    for operand in op.operands:
                        operand_str = str(operand)
                        print(operand_str)
                        input_types.append(str(operand.type))
                    args_str = ", ".join(
                        f"%arg{i}: {typ}" for i, typ in enumerate(input_types)
                    )
                    result_type = str(
                        op.result.type
                    )  # assuming there is only one return value
                    new_module_str = f"""module {{
                    func.func @main({args_str}) -> {result_type} {{
                        %0 = {str(op)}
                        return %0 : {result_type}
                    }}
                    }}"""
                    print(new_module_str)
                    breakpoint()
                    # print(op)
                    # print("\n")
                    # print(operand)
                    # print(type(operand))
                    # print(dir(operand))  # This will show all available attributes/methods

                    # # Try these too
                    # print(block.arguments)
                    # print(type(block.arguments))
                    # print(dir(block.arguments))

                    # # And maybe
                    # for i, arg in enumerate(block.arguments):
                    #     if arg == operand:
                    #         print(f"Found at position {i}")
                    # print("\n\n\n")

    return ops


# def get_ops_in_module(module: mlir.ir.Module) -> Dict[str, Dict[str, Any]]:
#     """
#     Get all operations in a module with their details.

#     Args:
#         module: MLIR module
#     Returns:
#         Dictionary mapping result names to operation details
#     """
#     ops = {}
#     result_counter = 0  # Counter for %0, %1, etc.

#     # Iterate through all functions in the module
#     for func_op in module.body.operations:
#         # Iterate through all blocks in the function
#         for block in func_op.regions[0].blocks:
#             # Iterate through all operations in the block
#             for op in block.operations:
#                 # Skip return operations and function declarations
#                 if not op.name.startswith(('func.', 'return')):
#                     # Get the operation arguments
#                     op_args = []
#                     for operand in op.operands:
#                         # If the operand is from another operation (starts with %)
#                         if hasattr(operand, 'owner'):
#                             if operand.owner is None:  # This is a block argument
#                                 arg_index = block.arguments.index(operand)
#                                 op_args.append(f"arg{arg_index}")
#                             else:
#                                 # Find which result of the owner operation this is
#                                 result_index = operand.owner.results.index(operand)
#                                 owner_index = list(op.owner.block.operations).index(operand.owner)
#                                 op_args.append(f"%{owner_index}")

#                     # Store operation details
#                     ops[f"%{result_counter}"] = {
#                         'op': op.name,
#                         'arguments': op_args
#                     }
#                     result_counter += 1

#     return ops


def analyze_shapes(module: mlir.ir.Module) -> Dict[str, Any]:
    """
    Analyze shapes of arguments and results in the module.

    Args:
        module: MLIR module
    Returns:
        Dictionary containing argument and result shapes
    """
    shapes = {"arguments": {}, "results": {}}

    # Iterate through functions to get arguments
    for func_op in module.body.operations:
        entry_block = func_op.regions[0].blocks[0]

        # Get argument shapes
        for i, arg in enumerate(entry_block.arguments):
            arg_name = f"arg{i}"
            shapes["arguments"][arg_name] = str(arg.type)

        # Get result shapes
        for block in func_op.regions[0].blocks:
            for op in block.operations:
                # Skip func and return ops
                if not op.name.startswith(("func.", "return")):
                    for i, result in enumerate(op.results):
                        result_name = f'%{len(shapes["results"])}'
                        shapes["results"][result_name] = str(result.type)

    return shapes


def analyze_module(module_str):
    """
    Analyze an MLIR module by traversing its operations and extracting shape information.

    Args:
        module_str: String containing MLIR code
    """
    # Parse the module
    ctx = Context()
    module = Module.parse(module_str, ctx)

    def get_shape_info(value_type):
        """Extract shape and element type information from an MLIR type"""
        if hasattr(value_type, "shape"):
            return {
                "shape": value_type.shape,
                "element_type": str(value_type.element_type),
            }
        return {"shape": None, "element_type": str(value_type)}

    def process_operation(op, depth=0):
        """Recursively process an operation and its regions"""
        breakpoint()
        indent = "  " * depth
        print(f"{indent}Operation: {op.name}")

        # Process operands (inputs)
        print(f"{indent}Operands:")
        for i, operand in enumerate(op.operands):
            shape_info = get_shape_info(operand.type)
            print(
                f"{indent}  {i}: Shape={shape_info['shape']}, Type={shape_info['element_type']}"
            )

        # Process results (outputs)
        print(f"{indent}Results:")
        for i, result in enumerate(op.results):
            shape_info = get_shape_info(result.type)
            print(
                f"{indent}  {i}: Shape={shape_info['shape']}, Type={shape_info['element_type']}"
            )

        # Process regions recursively
        for region in op.regions:
            for block in region.blocks:
                for nested_op in block.operations:
                    process_operation(nested_op, depth + 1)

    # Traverse the module
    for op in module.body.operations:
        breakpoint()
        process_operation(op)


# Example usage
mlir_code = """
module {
  func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
    return %2 : tensor<1x128xf32>
  }
}
"""


def analyze_function_args(module_str):
    """
    Specifically analyze function arguments in an MLIR module.

    Args:
        module_str: String containing MLIR code
    """
    ctx = Context()
    module = Module.parse(module_str, ctx)

    for op in module.body.operations:
        if op.name == "func.func":
            print(f"Function: {op.attributes['sym_name']}")

            # Process function arguments
            for i, arg in enumerate(op.regions[0].blocks[0].arguments):
                shape_info = get_shape_info(arg.type)
                print(f"  Argument {i}:")
                print(f"    Shape: {shape_info['shape']}")
                print(f"    Type: {shape_info['element_type']}")

            # Process function return types
            func_type = op.attributes["function_type"].value
            for i, return_type in enumerate(func_type.results):
                shape_info = get_shape_info(return_type)
                print(f"  Return {i}:")
                print(f"    Shape: {shape_info['shape']}")
                print(f"    Type: {shape_info['element_type']}")


# Example usage with specific function analysis
def main():
    module = get_module_from_str(mlir_code)
    ops = get_ops_in_module(module)
    print(ops)
    shapes = analyze_shapes(module)
    print(shapes)


if __name__ == "__main__":
    main()
