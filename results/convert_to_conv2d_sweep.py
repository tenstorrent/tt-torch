# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import argparse
import re

# Helper function to convert datatype strings to TTNN format
def convert_datatype_to_ttnn(datatype):
    if "bf16" in datatype:
        ttnn_dtype = "bfloat16"
    elif "f32" in datatype:
        ttnn_dtype = "float32"
    else:
        assert False, f"Unsupported datatype: {datatype}"
    return f"int(ttnn.{ttnn_dtype})"


# Helper function to get number of bytes per element based on datatype
def get_bytes_per_element(datatype):
    if "bfloat16" in datatype:
        return 2  # 2 bytes for bf16
    elif "float32" in datatype:
        return 4  # 4 bytes for f32
    else:
        return 2  # Default to 2 bytes if unknown


# Calculate tensor size in MB based on shape and datatype
def calculate_tensor_size_mb(shape, datatype):
    # Calculate number of elements in tensor
    num_elements = 1
    for dim in shape:
        num_elements *= dim

    # Get bytes per element based on datatype
    bytes_per_element = get_bytes_per_element(datatype)

    # Calculate size in MB
    size_bytes = num_elements * bytes_per_element
    size_mb = size_bytes / (1024 * 1024)

    return size_mb


# Helper function to parse array<i32> format strings into lists of integers
def parse_array_i32(array_str):
    return [
        int(dim)
        for dim in array_str.replace("array<i32:", "").replace(">", "").split(", ")
    ]


# Helper function to extract int value from string like "42 : i64"
def extract_int(value_str):
    return int(value_str.split(":")[0].strip())


# Helper function to handle layout configuration for input/weight/output
def get_layout_config(op, layout_type):

    # Determine the correct key and index based on layout_type
    if layout_type in ["input", "weight"]:
        key = "input_layouts"
        idx = 0 if layout_type == "input" else 1
    else:  # output
        key = "output_layouts"
        idx = 0

    # Extract layout configuration
    if key in op and len(op[key]) > idx:
        layout_info = op[key][idx]
        memory_config = layout_info.get("memory_config", [])

        if len(memory_config) >= 4:
            # Determine layout (TILE or ROW_MAJOR)
            layout = (
                "int(ttnn.TILE_LAYOUT)"
                if "tile" in str(memory_config[2])
                else "int(ttnn.ROW_MAJOR_LAYOUT)"
            )

            # Get memory location
            memory = f'"{memory_config[3]}"'

            # Convert datatype
            datatype = convert_datatype_to_ttnn(memory_config[2])

            return layout, memory, datatype
        else:
            assert (
                False
            ), f"Unsupported {layout_type}_memory_config, expecting 4 elements: {memory_config}"
    else:
        assert False, f"Unsupported {layout_type}_layout"


# Convert JSON format Conv2D ops to TTNN sweep entries.
def parse_json_to_sweep_entries(json_content):

    try:
        # Parse JSON content
        data = json.loads(json_content)

        # Ensure we have a list of ops
        ops = data if isinstance(data, list) else [data]

        entries = []
        for op in ops:
            # print("Processing op json:\n", json.dumps(op, indent=2))

            if op["name"] != "ttnn.conv2d":
                print(f"Skipping non-conv2d op: {op['name']}")
                continue

            if op["compilation_status"] != 6:
                print(f"Skipping compilation_status: {op['compilation_status']}")
                continue

            # Extract attributes
            attrs = op["attributes"]
            batch_size = extract_int(attrs["batch_size"])
            out_channels = extract_int(attrs["out_channels"])
            in_channels = extract_int(attrs["in_channels"])
            input_height = extract_int(attrs["input_height"])
            input_width = extract_int(attrs["input_width"])

            # Extract array dimensions
            kernel_dims = parse_array_i32(attrs["kernel_size"])
            kernel_height, kernel_width = kernel_dims[0], kernel_dims[1]

            stride_dims = parse_array_i32(attrs["stride"])
            stride_h, stride_w = stride_dims[0], stride_dims[1]

            padding_dims = parse_array_i32(attrs["padding"])
            pad_h, pad_w = padding_dims[0], padding_dims[1]

            dilation_dims = parse_array_i32(attrs["dilation"])
            dilation_h, dilation_w = dilation_dims[0], dilation_dims[1]

            # Extract groups
            groups = extract_int(attrs["groups"])

            # Get layout configurations
            input_layout, input_memory, input_datatype = get_layout_config(op, "input")
            weight_layout, weight_memory, weight_datatype = get_layout_config(
                op, "weight"
            )
            output_layout, output_memory, output_datatype = get_layout_config(
                op, "output"
            )

            # Calculate tensor sizes
            input_tensor_shape = [batch_size, in_channels, input_height, input_width]
            weight_tensor_shape = [
                out_channels,
                in_channels // groups,
                kernel_height,
                kernel_width,
            ]
            output_tensor_shape = [
                batch_size,
                out_channels,
                input_height,
                input_width,
            ]  # This is approximate

            input_size_mb = calculate_tensor_size_mb(input_tensor_shape, input_datatype)
            weight_size_mb = calculate_tensor_size_mb(
                weight_tensor_shape, weight_datatype
            )
            output_size_mb = calculate_tensor_size_mb(
                output_tensor_shape, output_datatype
            )

            total_io_size_mb = input_size_mb + weight_size_mb + output_size_mb

            # Check for bias (if there's a third input that's not a device)
            bias = False
            if "input_shapes" in op and len(op["input_shapes"]) > 2:
                third_input = op["input_shapes"][2]
                if "!ttnn.device" not in third_input:
                    bias = True
                    # Add bias tensor size if present
                    bias_tensor_shape = [out_channels]
                    bias_size_mb = calculate_tensor_size_mb(
                        bias_tensor_shape, input_datatype
                    )  # Assume same datatype as input
                    total_io_size_mb += bias_size_mb

            # Extract compile error if present
            compile_error = op.get("compile_error", "")

            # Extract model names if present
            model_names = op.get("model_names", [])

            # Create the sweep entry
            sweep_entry = [
                batch_size,
                out_channels,
                in_channels,
                input_height,
                input_width,
                kernel_height,
                kernel_width,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                groups,
                dilation_h,
                dilation_w,
                bias,
                [input_layout, input_memory, input_datatype],
                [weight_layout, weight_memory, weight_datatype],
                [output_layout, output_memory, output_datatype],
            ]

            entries.append((sweep_entry, compile_error, model_names, total_io_size_mb))

        return entries

    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return []


# Format a sweep entry as a string in the expected format, with preceding compile error if present.
def format_sweep_entry(
    entry, compile_error="", model_names=None, entry_number=0, io_size_mb=0
):

    # Add compile error as a comment if present
    # Add Entry number as comment
    if compile_error:
        error_lines = compile_error.strip().split("\n")
        result = f"# Testcase {entry_number}: {error_lines[0]}"
        for line in error_lines[1:]:
            # Skip empty lines and lines that are just "info:"
            if line.strip() == "" or line.strip() == "info:":
                continue
            result += f"\n# {line}"
        result += "\n"
    else:
        result = f"# Testcase {entry_number}\n"

    # Add model names as comment if present
    if model_names:
        formatted_name = (
            re.sub(r"(\d+):", r"(\1) ", model_names)
            if ":" in model_names
            else model_names
        )
        result += f"# Models: {formatted_name}\n"

    # Add I/O tensor size information
    exceeds_limit = io_size_mb > 60
    limit_warning = " - EXCEEDS 60MB LIMIT!" if exceeds_limit else ""
    result += f"# I/O Tensor Size: {io_size_mb:.2f} MB{limit_warning}\n"

    result += "["
    for i, item in enumerate(entry):
        if isinstance(item, list):
            sublist = "["
            for j, subitem in enumerate(item):
                sublist += str(subitem)
                if j < len(item) - 1:
                    sublist += ", "
            sublist += "]"
            result += sublist
        else:
            result += str(item)

        if i < len(entry) - 1:
            result += ", "

    result += "],"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert Conv2D ops from JSON to TTNN sweep entries"
    )
    parser.add_argument("--input_file", required=True, help="Input JSON file")
    parser.add_argument("--output_file", help="Output file for sweep entries")

    args = parser.parse_args()

    # Validate input file extension
    if not args.input_file.lower().endswith(".json"):
        print("Error: Input file must be a JSON file")
        return

    # Read input file
    with open(args.input_file, "r") as f:
        content = f.read()

    # Parse JSON content
    entries = parse_json_to_sweep_entries(content)

    if entries:
        print(f"Generated {len(entries)} TTNN sweep entries\n")

        formatted_entries = []
        for i, (entry, compile_error, model_names, io_size_mb) in enumerate(entries):
            formatted_entry = format_sweep_entry(
                entry, compile_error, model_names, i, io_size_mb
            )
            formatted_entries.append(formatted_entry)
            print(f"{formatted_entry}\n")

        if args.output_file:
            with open(args.output_file, "w") as f:
                for entry in formatted_entries:
                    f.write(entry + "\n")
            print(f"Sweep entries written to {args.output_file}")
    else:
        print("Failed to generate sweep entries.")


if __name__ == "__main__":
    main()
