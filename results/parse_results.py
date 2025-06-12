# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import json
import xlsxwriter


# Script to parse the results of the unique ops json files and combine them into a spreadsheet
# This script parses models compiled into stable hlo / TTIR in one shot (as opposed to op by op)
def find_json_files(directory="results"):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def extract_shape(shape_str):
    inputs = shape_str.split("_x_")
    ret = ""
    inp_strings = []
    for inp in inputs:
        dims = inp.split("x")
        dim_str = ",".join(dims)
        inp_strings.append(f"({dim_str})")
    return "x".join(inp_strings)


def process_json_files():
    json_files = find_json_files()

    ops_per_model = {}
    models_per_op = {}
    workbook = xlsxwriter.Workbook("results/models_per_op.xlsx")
    bold = workbook.add_format({"bold": True})
    for json_file in json_files:
        print(json_file)
        with open(json_file, "r") as f:
            data = json.load(f)

        model_name = json_file.strip("_unique_ops.json").split("/")[-1]
        ops_per_model[model_name] = list(data.keys())
        for key in data.keys():
            if key not in models_per_op:
                models_per_op[key] = []
            models_per_op[key].append(model_name)

        worksheet = workbook.add_worksheet(model_name)
        keys = list(data.keys())
        keys.sort()
        row = 0
        header = ("OpName", "Shape", "NumOps", "Ops")
        worksheet.write_row(row, 0, header, bold)
        row += 1
        for op_name in keys:
            op_name_written = False
            for shape in data[op_name].keys():
                num_ops = data[op_name][shape]["num_ops"]

                row_data = (
                    [op_name, extract_shape(shape), num_ops]
                    if not op_name_written
                    else ["", extract_shape(shape), num_ops]
                )
                op_name_written = True
                worksheet.write_row(row, 0, row_data)
                row += 1
                for op in data[op_name][shape]["ops"]:
                    worksheet.write(row, 3, op)
                    worksheet.set_row(row, None, None, {"hidden": True})
                    row += 1
                ops = data[op_name][shape]["ops"]
        worksheet.autofit()

    ops = list(models_per_op.keys())
    ops.sort()
    models = list(ops_per_model.keys())
    worksheet = workbook.add_worksheet("ModelsPerOp")
    worksheet.set_column(0, 0, 35)  # first column width
    header = ["op"]
    header.extend(models)

    row = 0
    worksheet.write_row(row, 0, header, bold)
    row += 1
    for op in ops:
        data = [op] + [1 if model in models_per_op[op] else 0 for model in models]
        worksheet.write_row(row, 0, data)
        row += 1

    workbook.close()


if __name__ == "__main__":
    process_json_files()
