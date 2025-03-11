# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys
import os
import json
import csv
import xlsxwriter
from mdutils.mdutils import MdUtils
from pathlib import Path

import subprocess
import re

# Script to parse the results of the unique ops json files and combine them into a spreadsheet
# This script parses models compiled into stable hlo / TTIR op by op
def find_json_files(directory="results"):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def generate_status_report():
    json_files = find_json_files()
    status_report = {}
    model_names = []

    for json_file in json_files:

        if (
            "_unique_ops.json" not in json_file
        ):  # indicator that the JSON is a model report, not an op report
            continue

        # Get model name from the parent directory of the json file
        model_name = (
            json_file.strip("_unique_ops.json")
            .split("/")[-1]
            .split(" ")[0]
            .split("test")[-1]
        )
        if len(model_name) > 28:
            model_name = model_name[:28]
        id = 1
        while model_name in model_names:
            model_name = model_name + f"_{id}"
            id += 1

        model_names.append(model_name)

        with open(json_file, "r") as f:
            data = json.load(f)

        # Initialize status counts
        status_counts = {i: 0 for i in range(1, 8)}  # Status 1 to 7

        # Count the occurrences of each status in the JSON file
        for value in data.values():
            status = value.get("compilation_status", None)
            if status in status_counts:
                status_counts[status] += 1

        total_ops = sum(status_counts.values())
        status_percentages = {}
        for status, count in status_counts.items():
            if total_ops > 0:
                if status == 5:
                    # Status 5 is the sum of status 1-5 divided by the sum of status 1-7
                    status_percentages[status] = (
                        sum(status_counts[i] for i in range(1, 6)) / total_ops * 100
                    )
                elif status == 6:
                    status_percentages[status] = count / total_ops * 100
                elif status == 7:
                    status_percentages[status] = count / total_ops * 100
                else:
                    # For other statuses, calculate as usual
                    status_percentages[status] = (
                        (count / total_ops) * 100 if total_ops > 0 else 0
                    )
        status_report[model_name] = status_percentages

    sorted_status_report = {
        model_name: status_percentages
        for model_name, status_percentages in sorted(
            status_report.items(), key=lambda item: item[0].lower()
        )
    }
    md_file = MdUtils(file_name="results/models.md")

    # Add a title
    md_file.new_header(level=1, title="Model Compilation Status Report")

    table_header = [
        "Model Name",
        "Doesn't Compile",
        "Compiles, Doesn't Execute",
        "Runs on Device",
    ]

    # Prepare rows data: start with the header row
    table_data = [table_header]

    for model, status_percentages in sorted_status_report.items():
        row_data = [model]
        for status in range(5, 8):
            row_data.append(f"{status_percentages.get(status, 0):.2f}%")
        table_data.append(row_data)

    # Flatten the table_data to create a single list of strings
    flat_table_data = [item for row in table_data for item in row]
    md_file.new_table(
        columns=len(table_header), rows=len(table_data), text=flat_table_data
    )
    md_file.create_md_file()


def extract_shape(shape_list):
    def append_shape(shape):
        string = ""
        if isinstance(shape, (list, tuple)):
            string += "("
            string += ",".join([str(dim) for dim in shape])
            string += ")"
        else:
            string += str(shape)
        return string

    shape_strs = []
    for shape in shape_list:
        shape_strs.append(append_shape(shape))
    return "x".join(shape_strs)


def extract_shapes_md(shape_list):
    shape_str = ""
    for shape in shape_list:
        if len(shape):
            shape_str += f"Tensor<[{','.join([str(dim) for dim in shape])}]>,<br>"
        else:
            shape_str += "Scalar,<br>"

    return shape_str


def parse_runtime_output(output):
    """
    Parse the runtime stack dump to extract the error message
    """
    error_message = re.search(r"Error.*?TT_FATAL.*?\n", output)
    if error_message is not None:
        error_message = error_message.group(0)
        info_available = re.search(r"\ninfo:\n.*?\n", output)
        if info_available:
            error_message += info_available.group(0)

        return error_message

    error_message = re.search(r"Error.*TT_THROW.*\n", output)
    if error_message is not None:
        error_message = error_message.group(0)
        info_available = re.search(r"\ninfo:\n.*\n", output)
        if info_available:
            error_message += info_available.group(0)

        return error_message

    return "Error message not extracted."


def create_test_dirs():
    Path("results/mlir_tests/torch_ir").mkdir(parents=True, exist_ok=True)
    Path("results/mlir_tests/ttir").mkdir(parents=True, exist_ok=True)
    Path("results/mlir_tests/stable_hlo").mkdir(parents=True, exist_ok=True)


def process_json_files():
    json_files = find_json_files()
    create_test_dirs()

    ops_per_model = {}
    stable_hlo_ops_per_model = {}
    stable_hlo_ops = {}
    models_per_op = {}
    stable_hlo_models_per_op = {}
    model_names = []
    all_ops = {}
    stable_hlo_ops_per_torch_op = {}
    workbook = xlsxwriter.Workbook("results/models_op_per_op.xlsx")
    bold = workbook.add_format({"bold": True})
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        first_value = next(iter(data.values()))
        if "model_name" in first_value and len(first_value["model_name"]) > 0:
            model_name = first_value["model_name"]
        else:
            model_name = (
                json_file.strip("_unique_ops.json")
                .split("/")[-1]
                .split(" ")[0]
                .split("test")[-1]
            )
        # If invalid excel char in name: []:*?/\, replace with _
        model_name = re.sub(r"[^a-zA-Z0-9_]", "_", model_name)
        if len(model_name) > 28:
            model_name = model_name[:28]

        id = 1
        test_name = model_name
        while model_name in model_names:
            model_name = test_name + f"_{id}"
            id += 1

        print(f"Processing {model_name}")
        model_names.append(model_name)
        worksheet = workbook.add_worksheet(model_name)
        keys = list(data.keys())
        keys.sort()
        row = 0
        # xlsxwriter fails to write anything after 'Compiled Json' field; so it
        # is being written to the last column.
        header = (
            "Torch Name",
            "Input Shapes",
            "Output Shapes",
            "NumOps",
            "Status",
            "PCC",
            "ATOL",
            "Ops",
            "Torch IR",
            "Raw SHLO",
            "Raw TTIR",
            "Raw TTNNIR",
            "Compile Error",
            "Trace dump",
            "Compiled Json",
        )
        worksheet.write_row(row, 0, header, bold)
        row += 1
        torch_ops = {}
        for key, value in data.items():
            if key not in all_ops:
                all_ops[key] = value

            if value["torch_name"] not in torch_ops:
                torch_ops[value["torch_name"]] = []

            torch_ops[value["torch_name"]].append(
                {
                    "torch_name": value["torch_name"],
                    "input_shapes": value["input_shapes"],
                    "output_shapes": value["output_shapes"],
                    "num_ops": value["num_ops"],
                    "status": value["compilation_status"],
                    "torch_ir_graph": value["torch_ir_graph"],
                    "stable_hlo_graph": value["stable_hlo_graph"],
                    "ops": value["stable_hlo_ops"],
                    "ttir_graph": value["ttir_graph"],
                    "ttnn_graph": value["ttnn_graph"],
                    "compiled_json": value["compiled_json"],
                    "runtime_stack_error": value["runtime_stack_dump"],
                    "key": key,
                    "pcc": value["pcc"],
                    "atol": value["atol"],
                }
            )
        ops_per_model[model_name] = list(torch_ops.keys())
        for key in torch_ops.keys():
            if key not in models_per_op:
                models_per_op[key] = []
            models_per_op[key].append(model_name)

        stable_hlo_ops_per_model[model_name] = set()
        for torch_name, torch_op in sorted(torch_ops.items()):
            stable_hlo_ops_per_torch_op[torch_name] = set()
            name = torch_name
            test_num = 0
            for op in torch_op:
                num_ops = op["num_ops"]
                input_shapes = extract_shape(op["input_shapes"])
                output_shapes = extract_shape(op["output_shapes"])
                status = op["status"]
                raw_shlo = op["stable_hlo_graph"]
                ops = op["ops"]
                error = ""
                trace_dump = ""
                pcc = op["pcc"]
                atol = op["atol"]

                if 2 <= status <= 5:
                    if 2 <= status <= 3:
                        # Does not compile to Torch Backend (status == 2)
                        # Does not compile to StableHLO (status == 3)
                        test_name = f"{torch_name}_{test_num}.mlir"
                        test_num += 1
                        filename = f"results/mlir_tests/torch_ir/{test_name}"
                        with open(filename, "w") as f:
                            f.write(op["torch_ir_graph"])
                        result = subprocess.run(
                            [
                                "python3",
                                "results/lower_to_stablehlo.py",
                                filename,
                            ],
                            capture_output=True,
                            text=True,
                        )
                    elif status == 4:
                        # Does not compile to TTIR, create unit test
                        test_name = f"{torch_name}_{test_num}.mlir"
                        test_num += 1
                        with open(
                            f"results/mlir_tests/stable_hlo/{test_name}", "w"
                        ) as f:
                            f.write(op["stable_hlo_graph"])

                        result = subprocess.run(
                            [
                                "ttmlir-opt",
                                "--stablehlo-to-ttir-pipeline=enable-remove-dead-values=true",
                                f"results/mlir_tests/stable_hlo/{test_name}",
                            ],
                            capture_output=True,
                            text=True,
                        )
                    elif status == 5:
                        # Does not compile to TTNNIR, create unit test
                        test_name = f"{torch_name}_{test_num}.mlir"
                        test_num += 1
                        with open(f"results/mlir_tests/ttir/{test_name}", "w") as f:
                            f.write(op["ttir_graph"])

                        result = subprocess.run(
                            [
                                "ttmlir-opt",
                                "--ttir-to-ttnn-backend-pipeline",
                                f"results/mlir_tests/ttir/{test_name}",
                            ],
                            capture_output=True,
                            text=True,
                        )
                    if result.returncode != 0:
                        error = result.stderr.split("\n")[0]
                        trace_dump = result.stderr
                elif status == 6:
                    trace_dump = op["runtime_stack_error"]
                    trace_dump = trace_dump.replace("\\n", "\n")
                    error = parse_runtime_output(trace_dump)
                    trace_dump = re.sub(r"[^\x20-\x7E]", "", trace_dump)

                row_data = [
                    name,
                    input_shapes,
                    output_shapes,
                    num_ops,
                    status,
                    pcc,
                    atol,
                    "",
                    op["torch_ir_graph"],
                    raw_shlo,
                    op["ttir_graph"],
                    op["ttnn_graph"],
                    error,
                    trace_dump,
                    op["compiled_json"],
                ]
                all_ops[op["key"]]["error"] = error
                all_ops[op["key"]]["trace_dump"] = trace_dump
                worksheet.write_row(row, 0, row_data)
                name = ""
                row += 1
                for shlo_op in ops:
                    if shlo_op[1] not in stable_hlo_ops:
                        stable_hlo_ops[shlo_op[1]] = []
                    op = shlo_op
                    op.append(torch_name)
                    op.append(input_shapes)
                    op.append(output_shapes)
                    op.append(status)
                    stable_hlo_ops[shlo_op[1]].append(op)
                    stable_hlo_ops_per_model[model_name].add(shlo_op[1])
                    stable_hlo_ops_per_torch_op[torch_name].add(shlo_op[1])
                    row_data = ["", "", "", "", "", shlo_op[-1]]
                    worksheet.write_row(row, 0, row_data)
                    worksheet.set_row(row, None, None, {"hidden": True})
                    row += 1
        for shlo_op in stable_hlo_ops_per_model[model_name]:
            if shlo_op not in stable_hlo_models_per_op:
                stable_hlo_models_per_op[shlo_op] = []
            stable_hlo_models_per_op[shlo_op].append(model_name)
        worksheet.autofit()

    row = 0
    unique_ops = set()
    worksheet = workbook.add_worksheet("All Ops")
    # xlsxwriter fails to write anything after 'Compiled Json' field; so it is
    # being written to the last column.
    header = (
        "Torch Name",
        "Input Shapes",
        "Output Shapes",
        "NumOps",
        "Status",
        "PCC",
        "ATOL",
        "Ops",
        "Torch IR",
        "Raw SHLO",
        "Raw TTIR",
        "Raw TTNNIR",
        "Compile Error",
        "Trace dump",
        "Compiled JSON",
    )
    worksheet.write_row(row, 0, header, bold)
    row += 1
    torch_ops = {}
    for key, value in sorted(all_ops.items()):
        if key in unique_ops:
            continue
        unique_ops.add(key)
        if value["torch_name"] not in torch_ops:
            torch_ops[value["torch_name"]] = []
        else:
            torch_ops[value["torch_name"]].append(
                {
                    "torch_name": value["torch_name"],
                    "input_shapes": value["input_shapes"],
                    "output_shapes": value["output_shapes"],
                    "num_ops": value["num_ops"],
                    "status": value["compilation_status"],
                    "pcc": value["pcc"],
                    "atol": value["atol"],
                    "torch_ir_graph": value["torch_ir_graph"],
                    "stable_hlo_graph": value["stable_hlo_graph"],
                    "ops": value["stable_hlo_ops"],
                    "ttir_graph": value["ttir_graph"],
                    "ttnn_graph": value["ttnn_graph"],
                    "compiled_json": value["compiled_json"],
                    "error": value["error"],
                    "trace_dump": value["trace_dump"],
                }
            )

    for torch_name, torch_op in sorted(torch_ops.items()):
        name = torch_name
        for op in torch_op:
            num_ops = op["num_ops"]
            input_shapes = extract_shape(op["input_shapes"])
            output_shapes = extract_shape(op["output_shapes"])
            status = op["status"]
            pcc = op["pcc"]
            atol = op["atol"]
            torch_ir_graph = op["torch_ir_graph"]
            raw_shlo = op["stable_hlo_graph"]
            ops = op["ops"]
            ttir_graph = op["ttir_graph"]
            ttnn_graph = op["ttnn_graph"]
            compiled_json = op["compiled_json"]
            error = op["error"]
            trace_dump = op["trace_dump"]
            row_data = [
                name,
                input_shapes,
                output_shapes,
                num_ops,
                status,
                pcc,
                atol,
                "",
                torch_ir_graph,
                raw_shlo,
                ttir_graph,
                ttnn_graph,
                error,
                trace_dump,
                compiled_json,
            ]
            worksheet.write_row(row, 0, row_data)
            row += 1
            for shlo_op in ops:
                row_data = ["", "", "", "", "", shlo_op[-1]]
                worksheet.write_row(row, 0, row_data)
                worksheet.set_row(row, None, None, {"hidden": True})
                row += 1

    # worksheet.autofit()

    ops = list(models_per_op.keys())
    ops.sort()
    models = list(ops_per_model.keys())
    worksheet = workbook.add_worksheet("AtenModelsPerOp")
    row = 0
    row_data = ["Total Ops", len(ops)]
    worksheet.write_row(row, 0, row_data, bold)
    row += 1
    row_data = ["Total Models", len(models)]
    worksheet.write_row(row, 0, row_data, bold)
    row += 1
    worksheet.set_column(0, 0, 35)  # first column width
    header = ["op"]
    header.extend(models)

    worksheet.write_row(row, 0, header, bold)
    row += 1
    for op in ops:
        data = [op] + [1 if model in models_per_op[op] else 0 for model in models]
        worksheet.write_row(row, 0, data)
        row += 1

    ops = list(stable_hlo_models_per_op.keys())
    ops.sort()
    models = list(stable_hlo_ops_per_model.keys())
    worksheet = workbook.add_worksheet("StableHLOModelsPerOp")
    row = 0
    row_data = ["Total Ops", len(ops)]
    worksheet.write_row(row, 0, row_data, bold)
    row += 1
    row_data = ["Total Models", len(models)]
    worksheet.write_row(row, 0, row_data, bold)
    row += 1
    worksheet.set_column(0, 0, 35)  # first column width
    header = ["op"]
    header.extend(models)
    worksheet.write_row(row, 0, header, bold)
    row += 1
    for op in ops:
        data = [op] + [
            1 if model in stable_hlo_models_per_op[op] else 0 for model in models
        ]
        worksheet.write_row(row, 0, data)
        row += 1

    torch_ops = list(stable_hlo_ops_per_torch_op.keys())
    torch_ops.sort()

    shlo_ops = list(stable_hlo_models_per_op.keys())
    shlo_ops.sort()

    row = 0
    worksheet = workbook.add_worksheet("StableHLOOpssPerTorchOp")
    row_data = ["Total Torch Ops", len(torch_ops)]
    worksheet.write_row(row, 0, row_data, bold)
    row += 1
    row_data = ["Total StableHLO Ops", len(shlo_ops)]
    worksheet.write_row(row, 0, row_data, bold)
    row += 1
    worksheet.set_column(0, 0, 35)  # first column width
    header = ["op"] + [shlo_op.split(".")[1] for shlo_op in shlo_ops]
    worksheet.write_row(row, 0, header, bold)
    row += 1
    for torch_op in torch_ops:
        data = [torch_op] + [
            1 if shlo_op in stable_hlo_ops_per_torch_op[torch_op] else 0
            for shlo_op in shlo_ops
        ]
        worksheet.write_row(row, 0, data)
        row += 1

    workbook.close()

    op_mappings = {
        # "arith.constant": "ttnn.",
        "stablehlo.abs": "ttnn.abs",
        "stablehlo.add": "ttnn.add",
        "stablehlo.and": "ttnn.and",
        # "stablehlo.broadcast_in_dim": "ttnn.",
        "stablehlo.ceil": "ttnn.ceil",
        "stablehlo.clamp": "ttnn.clamp",
        "stablehlo.compare": "ttnn.?",
        "stablehlo.concatenate": "ttnn.concat",
        # "stablehlo.constant": "ttnn.",
        # "stablehlo.convert": "ttnn.",
        "stablehlo.convolution": "ttnn.conv2d",
        "stablehlo.cosine": "ttnn.cos",
        "stablehlo.divide": "ttnn.div",
        "stablehlo.dot_general": "ttnn.matmul",
        "stablehlo.dynamic_iota": "ttnn.arange",
        "stablehlo.exponential": "ttnn.exp",
        "stablehlo.floor": "ttnn.floor",
        "stablehlo.gather": "ttnn.embedding",
        "stablehlo.iota": "ttnn.arange",
        "stablehlo.log": "ttnn.log",
        "stablehlo.logistic": "ttnn.sigmoig",
        "stablehlo.maximum": "ttnn.maximum",
        "stablehlo.minimum": "ttnn.minimum",
        "stablehlo.multiply": "ttnn.multiply",
        "stablehlo.negate": "ttnn.neg",
        "stablehlo.not": "ttnn.not",
        "stablehlo.or": "ttnn.or",
        "stablehlo.power": "ttnn.pow",
        "stablehlo.reduce_stablehlo.add": "ttnn.sum",
        "stablehlo.reduce_stablehlo.and": "ttnn.?",
        "stablehlo.reduce_stablehlo.maximum": "ttnn.max",
        "stablehlo.reduce_stablehlo.or": "ttnn.?",
        "stablehlo.reduce_window_stablehlo.add": "ttnn.avg_pool2d",
        "stablehlo.remainder": "ttnn.remainder",
        "stablehlo.reshape": "ttnn.reshape",
        "stablehlo.reverse": "ttnn.?",
        "stablehlo.rsqrt": "ttnn.rsqrt",
        "stablehlo.scatter": "ttnn.scatter",
        "stablehlo.select": "ttnn.where",
        "stablehlo.sine": "ttnn.sin",
        "stablehlo.slice": "ttnn.slice",
        "stablehlo.sqrt": "ttnn.sqrt",
        "stablehlo.subtract": "ttnn.subtract",
        "stablehlo.tanh": "ttnn.tanh",
        "stablehlo.transpose": "ttnn.permute",
        # "tensor.empty": "ttnn.",
    }

    def process_compare(shlo_op, md_data):
        inputs_and_attr = extract_shapes_md(shlo_op[4])
        md_data.append(inputs_and_attr)
        op = shlo_op[2][0]
        if op == "EQ":
            md_data.append("ttnn.eq")
        elif op == "GT":
            md_data.append("ttnn.gt")
        elif op == "LT":
            md_data.append("ttnn.lt")
        elif op == "NE":
            md_data.append("ttnn.ne")
        elif op == "GE":
            md_data.append("ttnn.ge")

        md_data.append(shlo_op[7])
        md_data.append(f"{shlo_op[10]}")
        return

    def process_gather(shlo_op, md_data):
        # '%0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4096>}> : (tensor<32000x4096xbf16>, tensor<1x32xi64>) -> tensor<1x32x4096xbf16>'
        shlo_attr = [
            "offset_dims",
            "collapsed_slice_dims",
            "start_index_map",
            "index_vector_dim",
            "indices_are_sorted",
            "slice_sizes",
        ]
        terminators = [",", "}", ">", ":"]
        inputs_and_attr = extract_shapes_md(shlo_op[4])
        shlo = shlo_op[6]
        for attr in shlo_attr:
            if attr in shlo:
                res = shlo.split(f"{attr} = ")[1]
                open_brackets = 0
                for idx, char in enumerate(res):
                    if char == "[":
                        open_brackets += 1
                    elif char == "]":
                        open_brackets -= 1
                    if open_brackets == 0 and char in terminators:
                        break
                res = res[:idx]
                inputs_and_attr += f"{attr}: {res}<br>"

        md_data.append(inputs_and_attr)
        md_data.append("ttnn.embedding")

        md_data.append(shlo_op[7])
        md_data.append(f"{shlo_op[10]}")

        return

    def process_convolution(shlo_op, md_data):
        # {'(%arg0, %arg1) dim_numbers': '[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]', 'window': '{stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64}'}
        shlo_attr = [
            "stride",
            "pad",
            "rhs_dilate",
            "lhs_dilate",
            "batch_group_count",
            "feature_group_count",
        ]
        terminators = [",", "}", ":"]
        inputs_and_attr = extract_shapes_md(shlo_op[4])
        shlo = shlo_op[6]
        for attr in shlo_attr:
            if attr in shlo:
                res = shlo.split(f"{attr} = ")[1]
                open_brackets = 0
                for idx, char in enumerate(res):
                    if char == "[":
                        open_brackets += 1
                    elif char == "]":
                        open_brackets -= 1
                    if open_brackets == 0 and char in terminators:
                        break
                res = res[:idx]
                inputs_and_attr += f"{attr}: {res}<br>"

        md_data.append(inputs_and_attr)
        md_data.append("ttnn.conv2d")
        md_data.append(shlo_op[7])
        md_data.append(f"{shlo_op[10]}")

        return

    def process_reshape(shlo_op, md_data):
        # %1 = stablehlo.reshape %0 : (tensor<1x32x32xf32>) -> tensor<1x32x32x1xf32>
        inputs_and_attr = extract_shapes_md(shlo_op[4])
        inputs_and_attr += extract_shapes_md(shlo_op[5])

        md_data.append(inputs_and_attr)
        md_data.append("ttnn.reshape")
        md_data.append(shlo_op[7])
        md_data.append(f"{shlo_op[10]}")

        return

    def process_scatter(shlo_op, md_data):
        # %1 = "stablehlo.scatter"(%arg0, %c, %0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):stablehlo.return %arg3 : tensor<bf16>}) : (tensor<1x3x720x1280xbf16>, tensor<1x1xi64>, tensor<1x3x720x1280xbf16>) -> tensor<1x3x720x1280xbf16>
        shlo_attr = [
            "update_window_dims",
            "inserted_window_dims",
            "scatter_dims_to_operand_dims",
            "index_vector_dim",
        ]
        terminators = [",", "}", ":"]
        inputs_and_attr = extract_shapes_md(shlo_op[4])
        shlo = shlo_op[6]
        for attr in shlo_attr:
            if attr in shlo:
                res = shlo.split(f"{attr} = ")[1]
                open_brackets = 0
                for idx, char in enumerate(res):
                    if char == "[":
                        open_brackets += 1
                    elif char == "]":
                        open_brackets -= 1
                    if open_brackets == 0 and char in terminators:
                        break
                res = res[:idx]
                inputs_and_attr += f"{attr}: {res}<br>"

        md_data.append(inputs_and_attr)
        md_data.append("ttnn.scatter")
        md_data.append(shlo_op[7])
        md_data.append(f"{shlo_op[10]}")

        return

    def process_slice(shlo_op, md_data):
        #% 0 = stablehlo.slice %arg0 [0:1, 0:32, 0:32, 0:64] : (tensor<1x32x32x128xbf16>) -> tensor<1x32x32x64xbf16>
        inputs_and_attr = extract_shapes_md(shlo_op[4])
        indices = shlo_op[6][shlo_op[6].find("[") : shlo_op[6].find("]") + 1]
        inputs_and_attr += f"indices: {indices}<br>"

        md_data.append(inputs_and_attr)
        md_data.append("ttnn.reshape")
        md_data.append(shlo_op[7])
        md_data.append(f"{shlo_op[10]}")

    def default(shlo_op, md_data):
        if len(shlo_op[4]) == 1 and len(shlo_op[2]) != 1:
            shlo_op[4].append(shlo_op[4][0])
        inputs_and_attr = extract_shapes_md(shlo_op[4])
        attrs = shlo_op[3]
        for k, v in attrs.items():
            inputs_and_attr += f"{k}: {v}<br>"

        md_data.append(inputs_and_attr)
        if shlo_op[1] in op_mappings:
            md_data.append(op_mappings[shlo_op[1]])
        else:
            md_data.append("")

        md_data.append(shlo_op[7])
        md_data.append(f"{shlo_op[10]}")
        return

    workbook = xlsxwriter.Workbook("results/stable_hlo_ops.xlsx")
    keys = sorted(stable_hlo_ops.keys())
    for op in keys:
        md_file = MdUtils(file_name="docs/ops/" + op + ".md", title=op)
        md_file.create_md_file()

        title = op
        if op in op_mappings:
            title += "::" + op_mappings[op]

        md_file.new_header(level=3, title=title, add_table_of_contents="n")
        worksheet = workbook.add_worksheet(op[:31])

        row = 0
        header = (
            "Output",
            "Op Name",
            "Args",
            "Attrs",
            "Input Shapes",
            "Output Shapes",
            "HLO",
            "TorchName",
            "TorchIn",
            "TorchOut",
            "Status",
        )
        worksheet.write_row(row, 0, header, bold)
        row += 1
        md_data = ["", "STABLE HLO Input Variations", "ttnn op", "Torch Name", "Status"]
        unique_ops = set()
        index = 0
        for shlo_op in stable_hlo_ops[op]:
            md_data.append(f"{index}")
            if op == "stablehlo.compare":
                process_compare(shlo_op, md_data)
            elif op == "stablehlo.gather":
                process_gather(shlo_op, md_data)
            elif op == "stablehlo.convolution":
                process_convolution(shlo_op, md_data)
            elif op == "stablehlo.reshape":
                process_reshape(shlo_op, md_data)
            elif op == "stablehlo.slice":
                process_slice(shlo_op, md_data)
            elif op == "stablehlo.scatter":
                process_scatter(shlo_op, md_data)
            else:
                default(shlo_op, md_data)

            if md_data[-4] in unique_ops:
                md_data = md_data[:-5]
            else:
                index += 1
                unique_ops.add(md_data[-4])
            row_data = [elem.__str__() for elem in shlo_op]
            worksheet.write_row(row, 0, row_data)
            row += 1
        md_file.new_line()

        md_file.new_table(
            columns=5, rows=len(md_data) // 5, text=md_data, text_align="left"
        )
        md_file.new_line()
        text = md_file.file_data_text
        with open("docs/ops/" + op + ".md", "w") as f:
            f.write(text)

        worksheet.autofit()
    workbook.close()


if __name__ == "__main__":
    generate_status_report()
    process_json_files()
