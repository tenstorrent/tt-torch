# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from TTNNOps import TTNNOps
import re
import os
import getpass

#########################################################
# Helper functions
#########################################################


def convert_tensor_format(input_str):
    """
    Converts tensor type strings from one format to another.
    If input is not a tensor type, returns the input unchanged.
    Removes layout information if present.

    Args:
        input_str: String like 'tensor<1x128x64x64xbf16, #layout3>' or '!tt.device<#device>'

    Returns:
        Converted string for tensor types, original string for other types
    """
    # Check if it starts with 'tensor<'
    if not input_str.startswith("tensor<"):
        # If there's a layout in non-tensor type, remove it
        if ", #layout" in input_str:
            base = input_str.split(",")[0]
            return f"{base}>"
        return input_str

    # Extract the content inside the angle brackets
    start = input_str.find("<") + 1
    end = input_str.find(">")
    content = input_str[start:end]

    # Remove layout information by splitting on comma and taking first part
    content = content.split(",")[0].strip()

    # Split dimensions and data type
    dimensions = content.rsplit("x", 1)
    dims = dimensions[0].split("x")
    dtype = dimensions[1]

    # Format dimensions with brackets and include dtype
    formatted_content = f"[{','.join(dims)},{dtype}]"

    # Construct the output string
    result = f"tensor<{formatted_content}>"

    return result


#########################################################
# AllOps class inherits from TTNNOps and reorganizes ops
#########################################################


class AllOps:
    def __init__(self, input_dir):
        self.ops = {}
        self.parse_input_dir(input_dir)

    def parse_input_dir(self, input_dir):
        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                input_file_path = os.path.join(input_dir, filename)
                self.process_ops(input_file_path)

    def process_ops(self, input_file_path):
        with open(input_file_path, "r") as file:
            ttnn_code = file.read()
        ttnn_parser = TTNNOps(ttnn_code)
        for op in ttnn_parser.ops:
            input_shapes = []
            for elem in op["input_shapes"]:
                input_shapes.append(convert_tensor_format(elem))
                print(elem)
                print(input_shapes[-1])
            output_shapes = []
            for elem in op["output_shapes"]:
                output_shapes.append(convert_tensor_format(elem))
            opToWrite = {
                "name": op["name"],
                "attributes": op["attributes"],
                "input_shapes": input_shapes,
                "output_shapes": output_shapes,
            }
            input_layouts = []
            if op["input_shapes"] is not None:
                for i_shape in op["input_shapes"]:
                    if "layout" in i_shape:
                        match = re.search(r"#layout\d*", i_shape)
                        layout_id = match.group(0) if match else None
                        input_layout = ttnn_parser.layouts[layout_id]
                        layout = {
                            "mapping_from": input_layout.mapping_from,
                            "mapping_to": input_layout.mapping_to,
                            "memory_config": input_layout.memory_config,
                        }
                        input_layouts.append(layout)
            output_layouts = []
            if op["output_shapes"] is not None:
                for o_shape in op["output_shapes"]:
                    if "layout" in o_shape:
                        match = re.search(r"#layout\d*", o_shape)
                        layout_id = match.group(0) if match else None
                        output_layout = ttnn_parser.layouts[layout_id]
                        layout = {
                            "mapping_from": output_layout.mapping_from,
                            "mapping_to": output_layout.mapping_to,
                            "memory_config": output_layout.memory_config,
                        }
                        output_layouts.append(layout)
            opToWrite["input_layouts"] = input_layouts
            opToWrite["output_layouts"] = output_layouts
            if self.ops.get(opToWrite["name"]) is None:
                self.ops[opToWrite["name"]] = [opToWrite]
            else:
                if opToWrite not in self.ops[opToWrite["name"]]:
                    self.ops[opToWrite["name"]].append(opToWrite)

    def print_all_ops(self, outdir):
        if os.path.isdir(outdir):
            output_file_path = os.path.join(outdir, "output.txt")
        with open(output_file_path, "w") as file:
            for op_name, ops_list in self.ops.items():
                print(op_name)
                file.write(f"{op_name}:\n")
                for sub_dict in ops_list:
                    file.write(f"{sub_dict}\n")

    def create_md_files(self, output_dir):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        for key, dict_list in self.ops.items():
            # Create a Markdown file for each key
            file_path = os.path.join(output_dir, f"{key}.md")
            with open(file_path, "w") as file:
                file.write(f"# {key}\n\n")

                if dict_list:
                    # Write the table header
                    file.write(
                        "| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts |\n"
                    )
                    file.write(
                        "|------|--------------|---------------|------------|---------------|----------------|\n"
                    )

                    # Write each dictionary in the array to the table
                    for item in dict_list:
                        name = item.get("name", "")

                        # Join shapes with <br> for line breaks
                        input_shapes = " <br> ".join(item.get("input_shapes", []))

                        # For input layouts (array of dicts)
                        input_layouts = item.get("input_layouts", [])
                        input_layouts_str = " <br> ".join(
                            f"mapping_from: {d.get('mapping_from', '')}, mapping_to: {d.get('mapping_to', '')}, memory_config: {d.get('memory_config', '')}"
                            for d in input_layouts
                        )

                        # Join attributes with <br> for line breaks
                        attributes = " <br> ".join(
                            f"{k}: {v}" for k, v in item.get("attributes", {}).items()
                        )

                        # For output layouts (array of dicts)
                        output_layouts = item.get("output_layouts", [])
                        output_layouts_str = " <br> ".join(
                            f"mapping_from: {d.get('mapping_from', '')}, mapping_to: {d.get('mapping_to', '')}, memory_config: {d.get('memory_config', '')}"
                            for d in output_layouts
                        )

                        # Join output shapes with <br> for line breaks
                        output_shapes = " <br> ".join(item.get("output_shapes", []))

                        file.write(
                            f"| {name} | {input_shapes} | {input_layouts_str} | {attributes} | {output_shapes} | {output_layouts_str} |\n"
                        )


user = getpass.getuser()
TTNNOpExamples = "/localdev/" + user + "/tt-torch/tt_torch/tools/AllOps/"
resultsDir = "/localdev/" + user + "/tt-torch/tt_torch/tools/AllOps/"  # optional
mdDir = "/localdev/" + user + "/tt-torch/docs/ops/ttnn"
myOps = AllOps(TTNNOpExamples)
myOps.print_all_ops(resultsDir)  # optional
myOps.create_md_files(mdDir)
