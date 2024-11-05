# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys
import os
import json
import csv
import xlsxwriter
from mdutils.mdutils import MdUtils

# Script to parse the results of the unique ops json files and combine them into a spreadsheet
# This script parses models compiled into stable hlo / TTIR op by op
def find_json_files(directory="results"):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


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


def process_json_files():
    json_files = find_json_files()

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
        print(json_file)
        with open(json_file, "r") as f:
            data = json.load(f)

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
        worksheet = workbook.add_worksheet(model_name)
        keys = list(data.keys())
        keys.sort()
        row = 0
        header = (
            "Torch Name",
            "Input Shapes",
            "Output Shapes",
            "NumOps",
            "Status",
            "Ops",
            "Raw SHLO",
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
                    "stable_hlo_graph": value["stable_hlo_graph"],
                    "ops": value["stable_hlo_ops"],
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
            for op in torch_op:
                num_ops = op["num_ops"]
                input_shapes = extract_shape(op["input_shapes"])
                output_shapes = extract_shape(op["output_shapes"])
                status = op["status"]
                raw_shlo = op["stable_hlo_graph"]
                ops = op["ops"]
                row_data = [name, input_shapes, output_shapes, num_ops, status]
                worksheet.write_row(row, 0, row_data)
                name = ""
                row += 1
                row_data = ["", "", "", "", "", raw_shlo]
                worksheet.write_row(row, 0, row_data)
                worksheet.set_row(row, None, None, {"hidden": True})
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
    header = (
        "Torch Name",
        "Input Shapes",
        "Output Shapes",
        "NumOps",
        "Status",
        "Ops",
        "Raw SHLO",
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
                    "stable_hlo_graph": value["stable_hlo_graph"],
                    "ops": value["stable_hlo_ops"],
                }
            )

    for torch_name, torch_op in sorted(torch_ops.items()):
        name = torch_name
        for op in torch_op:
            num_ops = op["num_ops"]
            input_shapes = extract_shape(op["input_shapes"])
            output_shapes = extract_shape(op["output_shapes"])
            status = op["status"]
            raw_shlo = op["stable_hlo_graph"]
            ops = op["ops"]
            row_data = [name, input_shapes, output_shapes, num_ops, status]
            name = ""
            worksheet.write_row(row, 0, row_data)
            row += 1
            row_data = ["", "", "", "", "", raw_shlo]
            worksheet.write_row(row, 0, row_data)
            worksheet.set_row(row, None, None, {"hidden": True})
            row += 1
            for shlo_op in ops:
                row_data = ["", "", "", "", "", shlo_op[-1]]
                worksheet.write_row(row, 0, row_data)
                worksheet.set_row(row, None, None, {"hidden": True})
                row += 1

    worksheet.autofit()

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
    process_json_files()
