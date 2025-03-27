# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Operations Parser and Markdown/JSON Generator

This script parses TTNN operation information from Excel or JSON files
and generates documentation in Markdown and JSON formats. It supports:
- Parsing operation details from spreadsheets or JSON files
- Converting formats
- Creating documentation with operation attributes, shapes, and layouts
"""

from TTNNOps import TTNNOps
import re
import os
import pandas as pd
import argparse
import json

#########################################################
# Helper functions
#########################################################


def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f


def validate_directory(directory):
    if not os.path.exists(directory):
        raise argparse.ArgumentTypeError(f"{directory} does not exist")

    if not os.path.isdir(directory):
        raise argparse.ArgumentTypeError(f"{directory} is not a directory")

    return directory


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
        if ", #ttnn_layout" in input_str:
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
    def __init__(self):
        self.ops = {}
        self.do_assert = False

    def parse_json(self, json_path):
        with open(json_path, "r") as file:
            json_string = json.load(file)
            if isinstance(json_string, dict):
                ajs = json_string
            elif isinstance(json_string, str):
                ajs = json.loads(json_string)
            else:
                raise ValueError("Invalid json format")
            # is there any case we index anything besides 0th programs?
            # if yes, this needs to be revised
            # this indexing hasn't been tested with many json files
            ttnn_mlir = ajs["programs"][0]["debug_info"]["mlir"]["source"]
            pcc = "N/A"
            atol = "N/A"
            self.process_ops(ttnn_mlir, pcc, atol)

    def parse_xlsx(self, excel_path, failures_only):
        """
        Parse the 'All Ops' sheet from an Excel file, extracting operation details.

        Args:
            excel_path (str): Path to the Excel file containing operation data
        """
        # Read only the 'All Ops' sheet specifically
        df = pd.read_excel(excel_path, sheet_name="All Ops")

        # Validate required columns are present
        required_columns = ["Compile Error", "Raw TTNNIR", "Torch Name", "PCC", "ATOL"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing columns in 'All Ops' sheet: {missing_columns}")

        # Clean DataFrame, removing rows with missing Raw TTNNIR
        df_cleaned = df.dropna(subset=["Raw TTNNIR"])

        # Process each row in the cleaned DataFrame
        for index, row in df_cleaned.iterrows():
            # Remove quotes from Raw TTNNIR if present
            raw_ttnnir = row["Raw TTNNIR"].strip("'\"")

            # If outputing failures only, skip if row is empty, Nan, or "Error message not extracted."
            if failures_only:
                compile_error = row.get("Compile Error", "")
                if (
                    pd.isna(compile_error)
                    or str(compile_error).strip() == ""
                    or str(compile_error).strip() == "Error message not extracted."
                ):
                    continue

            # Extract row details
            pcc = row["PCC"]
            atol = row["ATOL"]

            # Process operation details
            self.process_ops(raw_ttnnir, pcc, atol)

    def process_ops(self, ttnnir_string, pcc, atol):
        """
        Process TTNN operations from an IR string, extracting shapes, layouts, and metadata.

        Args:
            ttnnir_string: TTNN Intermediate Representation string
            pcc: Percent Correct Classification metric
            atol: Absolute tolerance for numerical comparisons
        """

        ttnn_parser = TTNNOps(ttnnir_string)
        for op in ttnn_parser.ops:
            input_shapes = []
            for elem in op["input_shapes"]:
                input_shapes.append(convert_tensor_format(elem))
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
                        match = re.search(r"#ttnn_layout\d*", i_shape)
                        layout_id = match.group(0) if match else None
                        if layout_id in ttnn_parser.layouts:
                            input_layout = ttnn_parser.layouts[layout_id]
                            layout = {
                                "mapping_from": input_layout.mapping_from,
                                "mapping_to": input_layout.mapping_to,
                                "memory_config": input_layout.memory_config,
                            }
                        else:
                            if self.do_assert:
                                print(
                                    f"{op} using {layout_id} which is not defined in existing layouts: {ttnn_parser.layouts}\n"
                                )
                            layout = {
                                "mapping_from": "N/A",
                                "mapping_to": "N/A",
                                "memory_config": "N/A",
                            }
                        input_layouts.append(layout)
            output_layouts = []
            if op["output_shapes"] is not None:
                for o_shape in op["output_shapes"]:
                    if "layout" in o_shape:
                        match = re.search(r"#ttnn_layout\d*", o_shape)
                        layout_id = match.group(0) if match else None
                        if layout_id in ttnn_parser.layouts:
                            output_layout = ttnn_parser.layouts[layout_id]
                            layout = {
                                "mapping_from": output_layout.mapping_from,
                                "mapping_to": output_layout.mapping_to,
                                "memory_config": output_layout.memory_config,
                            }
                        else:
                            if self.do_assert:
                                print(
                                    f"{op} using {layout_id} which is not defined in existing layouts: {ttnn_parser.layouts}\n"
                                )
                            layout = {
                                "mapping_from": "N/A",
                                "mapping_to": "N/A",
                                "memory_config": "N/A",
                            }
                        output_layouts.append(layout)
            opToWrite["input_layouts"] = input_layouts
            opToWrite["output_layouts"] = output_layouts
            opToWrite["pcc"] = pcc
            opToWrite["atol"] = atol
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
                file.write(f"# {key}")
                if dict_list:
                    count = 0
                    # Write each dictionary in the array to the table
                    for item in dict_list:
                        if count % 600 == 0:
                            # Write the table header
                            file.write(
                                "\n\n| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts | PCC | ATOL |\n"
                            )
                            file.write(
                                "|------|--------------|---------------|------------|---------------|----------------|-----|------|\n"
                            )
                        name = item.get("name", "")
                        pcc = item.get("pcc", "")
                        atol = item.get("atol", "")

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
                            f"| {name} | {input_shapes} | {input_layouts_str} | {attributes} | {output_shapes} | {output_layouts_str} | {pcc} | {atol} |\n"
                        )
                        count += 1

    def create_json_data(self):
        # Create a dictionary to store JSON data for each key
        json_data = {}

        for key, dict_list in self.ops.items():
            # Create a list to store processed items for each key
            processed_items = []

            for item in dict_list:
                # Process input layouts
                pcc = "N/A" if pd.isna(item.get("pcc")) else str(item.get("pcc", "N/A"))
                atol = (
                    "N/A" if pd.isna(item.get("atol")) else str(item.get("atol", "N/A"))
                )

                input_layouts = item.get("input_layouts", [])
                processed_input_layouts = [
                    {
                        "mapping_from": d.get("mapping_from", ""),
                        "mapping_to": d.get("mapping_to", ""),
                        "memory_config": d.get("memory_config", ""),
                    }
                    for d in input_layouts
                ]

                # Process output layouts
                output_layouts = item.get("output_layouts", [])
                processed_output_layouts = [
                    {
                        "mapping_from": d.get("mapping_from", ""),
                        "mapping_to": d.get("mapping_to", ""),
                        "memory_config": d.get("memory_config", ""),
                    }
                    for d in output_layouts
                ]

                # Create a processed item dictionary
                processed_item = {
                    "name": item.get("name", ""),
                    "input_shapes": item.get("input_shapes", []),
                    "input_layouts": processed_input_layouts,
                    "attributes": item.get("attributes", {}),
                    "output_shapes": item.get("output_shapes", []),
                    "output_layouts": processed_output_layouts,
                    "pcc": pcc,
                    "atol": atol,
                }

                processed_items.append(processed_item)

            # Store processed items for each key
            json_data[key] = processed_items

        return json_data

    def save_json_files(self, output_dir):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get the JSON data
        json_data = self.create_json_data()

        # Save JSON files
        for key, data in json_data.items():
            file_path = os.path.join(output_dir, f"{key}.json")
            with open(file_path, "w") as file:
                json.dump(data, file, indent=2)
                file.write("\n")  # add a new line at the end of json

        return json_data


if __name__ == "__main__":
    """
    This script is used to create Markdown (.md) and JSON (.json) files
    based on the provided Excel (.xlsx) or JSON (.json) input files.

    Usage:
        python generate_md.py [OPTIONS]

    Options:
        --excel_path FILE
            The path to the models_op_per_op.xlsx file.
            Provide this option if you want to generate files based on an Excel input.

        --json_path FILE
            The path to the model.json file.
            Provide this option if you want to generate files based on a JSON input.

        --md_dir DIR
            The path to the directory where the generated Markdown files will be saved.

        --json_dir DIR
            The path to the directory where the generated JSON files will be saved.

        --failures_only
            Only output .json/.md results for ops that have legitimate errors.

    Examples:
        python generate_md.py --excel_path /path/to/models_op_per_op.xlsx --md_dir /path/to/md/output/
        python generate_md.py --json_path /path/to/model.json --json_dir /path/to/json/output/

    Notes:
        - You must provide either --excel_path and/or --json_path
    """
    parser = argparse.ArgumentParser(description="Create ttnn ops md files")
    parser.add_argument(
        "--excel_path",
        dest="excel_path",
        required=False,
        type=validate_file,
        help="the path to models_op_per_op.xlsx file",
        metavar="FILE",
    )
    parser.add_argument(
        "--json_path",
        dest="json_path",
        required=False,
        type=validate_file,
        help="the path to model.json file",
        metavar="FILE",
    )
    parser.add_argument(
        "--md_dir",
        dest="md_dir",
        required=False,
        type=validate_directory,
        help="the path to the directory where markdown files will be created.",
        metavar="DIR",
    )
    parser.add_argument(
        "--json_dir",
        dest="json_dir",
        required=False,
        type=validate_directory,
        help="the path to the directory where json files will be created.",
        metavar="DIR",
    )

    parser.add_argument(
        "--failures_only",
        dest="failures_only",
        action="store_true",
        help="Only output ops that have legitimate errors",
    )

    args = parser.parse_args()
    if args.excel_path is None and args.json_path is None:
        # if neither paths are provided
        print("Please provide either excel_path or json_path")
        exit(1)
    if args.excel_path is not None and args.json_path is not None:
        # if both paths are provided
        print("Please provide either excel_path or json_path")
        exit(1)

    if args.excel_path is not None:
        try:
            myOps = AllOps()
            myOps.parse_xlsx(args.excel_path, args.failures_only)
        except Exception as e:
            print(f"Exception occured at generate_md.py: {e}")
    if args.json_path is not None:
        try:
            myOps = AllOps()
            myOps.parse_json(args.json_path)
        except Exception as e:
            print(f"Exception occured at generate_md.py: {e}")

    if args.md_dir is None and args.json_dir is None:
        # if neither directories are provided
        print("Please provide either md_dir or json_dir")
        exit(1)

    if args.md_dir is not None:
        try:
            myOps.create_md_files(args.md_dir)
            print(f"Successfully generated .md files in {args.md_dir}")
        except Exception as e:
            print(f"Exception occured at generate_md.py: {e}")

    if args.json_dir is not None:
        try:
            myOps.save_json_files(args.json_dir)
            print(f"Successfully generated .json files in {args.json_dir}")
        except Exception as e:
            print(f"Exception occured at generate_md.py: {e}")
