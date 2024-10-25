import sys
import os
import json
import csv
import xlsxwriter

def find_json_files(directory = "results"):
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


def process_json_files():
    json_files = find_json_files()

    ops_per_model = {}
    stable_hlo_ops_per_model = {}
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

      model_name = json_file.strip("_unique_ops.json").split("/")[-1].split(" ")[0].split("test")[-1]
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
      header = ("Torch Name", "Input Shapes", "Output Shapes", "NumOps", "Status", "Ops", "Raw SHLO")
      worksheet.write_row(row, 0, header, bold)
      row += 1
      torch_ops = {}
      for key, value in data.items():
        if key not in all_ops:
          all_ops[key] = value

        if value["torch_name"] not in torch_ops:
          torch_ops[value["torch_name"]] = []

        torch_ops[value["torch_name"]].append({"torch_name":value["torch_name"], "input_shapes": value["input_shapes"], "output_shapes": value["output_shapes"], 
                                                "num_ops": value["num_ops"], "status": value["compilation_status"], "stable_hlo_graph": value["stable_hlo_graph"], 
                                                "ops": value["stable_hlo_ops"]})
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
    header = ("Torch Name", "Input Shapes", "Output Shapes", "NumOps", "Status", "Ops", "Raw SHLO")
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
        torch_ops[value["torch_name"]].append({"torch_name":value["torch_name"], "input_shapes": value["input_shapes"], "output_shapes": value["output_shapes"], 
                                                "num_ops": value["num_ops"], "status": value["compilation_status"], "stable_hlo_graph": value["stable_hlo_graph"], 
                                                "ops": value["stable_hlo_ops"]})

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
    worksheet.set_column(0, 0, 35) #first column width
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
    worksheet.set_column(0, 0, 35) #first column width
    header = ["op"]
    header.extend(models)
    worksheet.write_row(row, 0, header, bold)
    row += 1
    for op in ops:
      data = [op] + [1 if model in stable_hlo_models_per_op[op] else 0 for model in models]
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
    worksheet.set_column(0, 0, 35) #first column width
    header = ["op"] + [shlo_op.split(".")[1] for shlo_op in shlo_ops]
    worksheet.write_row(row, 0, header, bold)
    row += 1
    for torch_op in torch_ops:
      data = [torch_op] + [1 if shlo_op in stable_hlo_ops_per_torch_op[torch_op] else 0 for shlo_op in shlo_ops]
      worksheet.write_row(row, 0, data)
      row += 1


    workbook.close()

if __name__ == "__main__":
    process_json_files()