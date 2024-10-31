# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re
import json


def extract_shape(shape_str):
    if not shape_str.startswith("tensor<"):
        breakpoint()
    assert shape_str.startswith("tensor<")
    assert shape_str.endswith(">")
    shape_str = shape_str[len("tensor<") : -1]
    dims = shape_str.split("x")
    return [int(dim) for dim in dims[:-1]]


def split_top(string, splitter=",", openers="([{<", closers=")]}>", whitespace=" \n\t"):
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


def parse_mlir(mlir_code, verbose=False):
    ops = []
    unique_ops = {}
    for line in mlir_code.splitlines():
        line = line.strip()
        if not line.startswith("%"):
            continue
        if verbose:
            print(line)

        output = line.split(" = ")[0].strip()
        # if output == "%21":
        #   breakpoint()
        op_name = line.split(" = ")[1].split(" ")[0]
        if op_name.startswith('"'):
            op_name = op_name.split('"')[1]
        elif "(" in op_name:
            op_name = op_name.split("(")[0]
        if verbose:
            print(f"  op_name: {op_name}")
        # reduce is special cased
        args_and_attr = line.split(op_name)[1]
        if op_name == "stablehlo.reduce":
            op_name += "_" + args_and_attr.split("applies")[1].strip().split(" ")[0]
            dim = args_and_attr.split("dimensions = ")[1].split(" ")[0]
            attr = {"dim": dim}
            args = [args_and_attr.split(")")[0].strip("(")]
        else:
            args_and_attr = line.split(op_name)[1]
            args_and_attr = args_and_attr[: args_and_attr.rfind(":")]
            args_and_attr = split_top(args_and_attr)
            args = []
            attr = {}
            for arg in args_and_attr:
                if "=" in arg:
                    key, value = arg.split("=")
                    attr[key.strip()] = value.strip()
                else:
                    args.append(arg.strip())
            if verbose:
                print(f"  args: {args}")
                print(f"  attr: {attr}")
        io_shapes = line[line.rfind(":") + 1 :].strip()
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
        op = (output, op_name, args, attr, input_shapes, output_shapes, line)
        ops.append(op)

        if op_name not in unique_ops:
            unique_ops[op_name] = {}

        if len(input_shapes) == 0:
            key = ""
        else:
            key = print_shape(input_shapes[0])
        for shape in input_shapes[1:]:
            key += f"_x_{print_shape(shape)}"
        if key not in unique_ops[op_name]:
            unique_ops[op_name][key] = {}
            unique_ops[op_name][key]["ops"] = []
            unique_ops[op_name][key]["num_ops"] = 1
        else:
            unique_ops[op_name][key]["num_ops"] += 1
        unique_ops[op_name][key]["ops"].append(op)
    return ops, unique_ops
