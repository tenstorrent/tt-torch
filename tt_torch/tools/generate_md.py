# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from TTNNOps import TTNNOps
import re
import os
import pandas as pd
import argparse

#########################################################
# Helper functions
#########################################################
def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f


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
    def __init__(self):
        self.ops = {}
        self.do_assert = False

    def parse_xlsx(self, excel_path):
        # Read all sheets in the Excel file
        xls = pd.ExcelFile(excel_path)

        # Iterate through all sheets
        for sheet_name in xls.sheet_names:
            if sheet_name == "All Ops":
                continue
            # Read the current sheet
            df = pd.read_excel(excel_path, sheet_name=sheet_name)

            # Check if required columns exist
            required_columns = ["Raw TTNNIR", "Torch Name", "Status", "PCC", "ATOL"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(
                    f"Skipping sheet '{sheet_name}'. Missing columns: {missing_columns}"
                )
                continue

            # Clean the DataFrame
            df_cleaned = df.dropna(subset=["Raw TTNNIR"])
            df_final = df_cleaned[["Torch Name", "Raw TTNNIR", "Status", "PCC", "ATOL"]]

            # Check if DataFrame is empty after cleaning
            if df_final.empty and self.do_assert:
                print(f"Skipping sheet '{sheet_name}'. No valid data after cleaning.")
                continue

            # Initialize last torch name
            last_torch_name = None

            # Iterate through the DataFrame and write to files
            for index, row in df_final.iterrows():
                torch_name = (
                    row["Torch Name"]
                    if pd.notna(row["Torch Name"])
                    else last_torch_name
                )
                raw_ttnnir = row["Raw TTNNIR"]
                status = row["Status"]
                pcc = row["PCC"]
                atol = row["ATOL"]
                # Remove quotes if present
                if raw_ttnnir.startswith("'") and raw_ttnnir.endswith("'"):
                    raw_ttnnir = raw_ttnnir[1:-1]
                    self.process_ops(raw_ttnir, status, pcc, atol)
                elif raw_ttnnir.startswith('"') and raw_ttnnir.endswith('"'):
                    raw_ttnnir = raw_ttnnir[1:-1]
                    self.process_ops(raw_ttnir, status, pcc, atol)
                else:
                    self.process_ops(raw_ttnnir, status, pcc, atol)
                print(f"Finished {sheet_name}, {torch_name}")

    def process_ops(self, ttnnir_string, status, pcc, atol):
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
                        match = re.search(r"#layout\d*", i_shape)
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
                        match = re.search(r"#layout\d*", o_shape)
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
            if status == 6.0:
                opToWrite["runs_on_ttnn"] = "no"
            elif status == 7.0:
                opToWrite["runs_on_ttnn"] = "yes"
            else:
                opToWrite["runs_on_ttnn"] = "N/A"
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
                file.write(f"# {key}\n\n")

                if dict_list:
                    # Write the table header
                    file.write(
                        "| Name | Input Shapes | Input Layouts | Attributes | Output Shapes | Output Layouts | Runs on TTNN | PCC | ATOL |\n"
                    )
                    file.write(
                        "|------|--------------|---------------|------------|---------------|----------------|--------------|-----|------|\n"
                    )

                    # Write each dictionary in the array to the table
                    for item in dict_list:
                        name = item.get("name", "")
                        runs_on_ttnn = item.get("runs_on_ttnn", "")
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
                            f"| {name} | {input_shapes} | {input_layouts_str} | {attributes} | {output_shapes} | {output_layouts_str} | {runs_on_ttnn} | {pcc} | {atol} |\n"
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ttnn ops md files")
    parser.add_argument(
        "-i",
        "--excel_path",
        dest="excel_path",
        required=True,
        type=validate_file,
        help="the path to models_op_per_op.xlsx file",
        metavar="FILE",
    )
    args = parser.parse_args()
    print(args.excel_path)
    current_path = os.getcwd()
    mdDir = current_path + "/docs/ops/ttnn"
    try:
        myOps = AllOps()
        myOps.parse_xlsx(args.excel_path)
        myOps.create_md_files(mdDir)
    except Exception as e:
        print(f"Exception occured at generate_md.py: {e}")
