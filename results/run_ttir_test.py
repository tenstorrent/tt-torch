# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import glob
import os
import pandas as pd
import re
import shutil
import subprocess
import sys
import xlsxwriter

from pathlib import Path
from tt_torch.tools.utils import OpCompilationStatus


def check_compiler_requirements():
    """
    Check if all the required tools are available.
    """
    result_ttmlir = shutil.which("ttmlir-opt")
    if result_ttmlir is None:
        # raise FileNotFoundError(
        #   "ttmlir-opt not found. Please install tt-mlir compiler."
        # )
        print("ttmlir-opt not found", file=sys.stderr)

    result_ttrt = shutil.which("ttrt")
    if result_ttrt is None:
        # raise FileNotFoundError("ttrt not found. Please install tt-mlir compiler.")
        print("ttrt not found", file=sys.stderr)

    result_translate = shutil.which("ttmlir-translate")
    if result_translate is None:
        # raise FileNotFoundError(
        #   "ttmlir-translate not found. Please install tt-mlir compiler."
        # )
        print("ttmlir-translate not found", file=sys.stderr)


def create_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    Path(directory).mkdir(parents=True, exist_ok=True)


def generate_ttrt_artifacts(directory):
    """
    Generate TTRT artifacts for the current machine to be used by TTRT.
    """
    cmd = ["ttrt", "query", "--save-artifacts", "--artifact-dir", directory]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    return result


def run_ttir_to_ttnn_pipeline(file, artifact, output_dir):
    """
    Run TTIR to TTNN pipeline for TTIR graph with the generated TTRT artifact.
    Returns error code and stderr output.
    """
    cmd = [
        "ttmlir-opt",
        f"--ttir-to-ttnn-backend-pipeline=system-desc-path={artifact}",
        file,
    ]
    file_name = os.path.basename(file)
    ttnn_file = open(os.path.join(output_dir, file_name), "w")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        ttnn_file.write(result.stdout)
    ttnn_file.close()
    return result.returncode, result.stderr


def generate_flatbuffer(file, output_dir):
    """
    Generate flatbuffer binary file using TTNN graph.
    """
    file_name = os.path.basename(file)
    ttnn_file = os.path.join(output_dir, file_name)
    cmd = ["ttmlir-translate", "--ttnn-to-flatbuffer", ttnn_file]

    file_name = Path(file).stem
    output_file = open(os.path.join(output_dir, file_name + ".ttnn"), "wb")

    result = subprocess.run(cmd, stdout=output_file)


def execute_ttrt(file, output_dir):
    """
    Execute the flatbuffer binary with TTRT.
    """
    file_name = Path(file).stem
    fbb_file = os.path.join(output_dir, file_name + ".ttnn")
    cmd = ["ttrt", "run", fbb_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def parse_ttrt_output(result, filename, output_dir):
    """
    Parse the TTRT output and determine if a binary is successfully executed or
    failed.
    Returns 0 in case of success and 1 in case of error along with error message.
    """
    output = result.stderr
    filename = os.path.join(output_dir, filename + ".ttnn")
    pass_string = f"INFO - PASS: test case={filename}"

    if pass_string in output:
        return 0, ""

    error_message = re.search(r"ERROR - ERROR.*FATAL.*\n", output)
    if error_message is not None:
        error_message = error_message.group(0)
        info_available = re.search(r"\ninfo:\n.*\n", output)
        if info_available:
            error_message += info_available.group(0)

        return 1, error_message

    error_message = re.search(r"ERROR - ERROR.*TT_THROW.*\n", output)
    if error_message is not None:
        error_message = error_message.group(0)
        info_available = re.search(r"\ninfo:\n.*\n", output)
        if info_available:
            error_message += info_available.group(0)

        return 1, error_message

    return 1, "Error message not extracted."


def execute_ttir_test(file, ttrt_artifacts, output_dir):
    ttnn_errorcode, ttnn_error = run_ttir_to_ttnn_pipeline(
        file, ttrt_artifacts, output_dir
    )
    if ttnn_errorcode != 0:
        result = {
            "status": "TTIR->TTNN failure",
            "ttnn_error": ttnn_error,
            "ttrt_error": "",
            "ttrt_dump": "",
        }
        return result

    filename = Path(file).stem
    generate_flatbuffer(file, output_dir)
    ttrt_result = execute_ttrt(file, output_dir)
    ttrt_errorcode, ttrt_error = parse_ttrt_output(ttrt_result, filename, output_dir)

    ttrt_dump = ttrt_result.stdout + ttrt_result.stderr

    if ttrt_errorcode == 0:
        result = {
            "status": "TTRT success",
            "ttnn_error": "",
            "ttrt_error": "",
            "ttrt_dump": ttrt_dump,
        }
        return result

    result = {
        "status": "TTRT failure",
        "ttnn_error": "",
        "ttrt_error": ttrt_error,
        "ttrt_dump": ttrt_dump,
    }
    return result


def process_excel_sheet(excel_path, runner_idx, runner_total):
    dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(dir, "mlir_tests", "ttir_excel")
    output_dir = os.path.join(dir, "mlir_tests", "output", "ttir")
    ttrt_dir = os.path.join(dir, "mlir_tests", "output", "ttrt-artifacts")
    create_dir(input_dir)
    create_dir(output_dir)
    ttrt_output = generate_ttrt_artifacts(ttrt_dir)

    if ttrt_output.returncode != 0:
        raise FileNotFoundError("Failed to generate ttrt-artifacts")

    ttrt_artifacts = os.path.join(ttrt_dir, "system_desc.ttsys")

    # Read only the 'All Ops' sheet specifically
    df = pd.read_excel(excel_path, sheet_name="All Ops")
    # Validate required columns are present
    required_columns = ["Torch Name", "Input Shapes", "Status", "Raw TTIR"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in 'All Ops' sheet: {missing_columns}")

    torch_name = ""
    indices = df.index[df["Status"] == OpCompilationStatus.CONVERTED_TO_TTNN]
    ttrt_data = []
    ttrt_excel_path = os.path.join(
        Path(excel_path).parent, f"ttrt_runner_{runner_idx}.xlsx"
    )
    workbook = xlsxwriter.Workbook(ttrt_excel_path)
    bold = workbook.add_format({"bold": True})
    worksheet = workbook.add_worksheet("TTRT Result")
    xlsx_row = 0
    header = (
        "Index",
        "Status",
        "TTRT Error",
        "TTRT Dump",
    )
    worksheet.write_row(xlsx_row, 0, header, bold)
    xlsx_row += 1
    for idx in range(runner_idx, len(indices), runner_total):
        index = indices[idx]
        row = df.iloc[index]

        raw_ttir = row["Raw TTIR"].strip("'\"")
        if not pd.isna(row["Torch Name"]):
            torch_name = row["Torch Name"]

        test_name = torch_name + row["Input Shapes"] + ".mlir"
        test_path = os.path.join(input_dir, test_name)
        with open(test_path, "w") as f:
            f.write(raw_ttir)

        result = execute_ttir_test(test_path, ttrt_artifacts, output_dir)
        if result["status"] == "TTRT success":
            row_data = [index, OpCompilationStatus.EXECUTED, "", ""]
            worksheet.write_row(xlsx_row, 0, row_data)
        elif result["status"] == "TTRT failure":
            row_data = [
                index,
                OpCompilationStatus.CONVERTED_TO_TTNN,
                result["ttrt_error"],
                re.sub(r"[^\x20-\x7E]", "", result["ttrt_dump"]),
            ]
            worksheet.write_row(xlsx_row, 0, row_data)

        xlsx_row += 1

    workbook.close()


def update_model_file(dir, excel_path):
    ttrt_files = glob.glob(f"{dir}/ttrt_runner*")
    if len(ttrt_files) == 0:
        raise FileNotFoundError(f"No TTRT result sheet found in {dir}.")

    df = pd.read_excel(excel_path, sheet_name="All Ops")
    if "TTRT Error" not in df.columns:
        df["TTRT Error"] = ""
    if "TTRT Dump" not in df.columns:
        df["TTRT Dump"] = ""

    status_col = df.columns.get_loc("Status")
    ttrt_error_col = df.columns.get_loc("TTRT Error")
    ttrt_dump_col = df.columns.get_loc("TTRT Dump")

    for file in ttrt_files:
        ttrt_df = pd.read_excel(file, sheet_name="TTRT Result")
        for index, row in ttrt_df.iterrows():
            row_num = int(row["Index"])
            df.iat[row_num, status_col] = row["Status"]
            df.iat[row_num, ttrt_error_col] = row["TTRT Error"]
            df.iat[row_num, ttrt_dump_col] = row["TTRT Dump"]

    with pd.ExcelWriter(
        excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        # Write the TTRT results back to the existing sheet.
        df.to_excel(writer, sheet_name="All Ops", index=False)


if __name__ == "__main__":
    """
    This script is used to run TTIR graphs (extracted from models_op_by_op.xlsx
    Excel sheet) with TTRT and update the Excel sheet. It will update the status
    if TTRT executes the graph successfully; otherwise add TTRT error message
    and stack dump in the original sheet.
    This process is perfromed in two steps.
    1.  Extract the TTIR graph; execute them with TTRT; and store the TTRT
        results temporarily as Excel file(s). [Default option]
    2.  Update the Excel file with TTRT results. [Use '--update_model' option]

    Usage:
        python run_ttir_test.py [OPTIONS]

    Options:
        --runner_idx INTEGER
            Index of the current runner [Index starts with 0]. This option is
            used to distribute the workload between different runner and is
            currently used for github workflows.

        --runner_total INTEGER
            Total number of available runners. This option is used to distribute
            the workload between different runner and is currently used for
            github workflows.

        --update_model
            Update the original Excel sheet with TTRT results (Status, TTRT
            error, and TTRT stack dump).
    """
    parser = argparse.ArgumentParser(description="Execute TTIR graphs with TTRT")
    parser.add_argument(
        "--runner_idx",
        dest="runner_idx",
        default=0,
        required=False,
        type=int,
        help="Index of the current runner [starting with 0]; use to distribute the workload. Default: 0",
        metavar="INTEGER",
    )
    parser.add_argument(
        "--runner_total",
        dest="runner_total",
        default=1,
        required=False,
        type=int,
        help="Total number of runner for distribution of workload. Default: 1",
        metavar="INTEGER",
    )
    parser.add_argument(
        "--update_model",
        dest="update_model",
        action="store_true",
        help="Update models excel file with TTRT results.",
    )

    check_compiler_requirements()

    args = parser.parse_args()
    if args.runner_idx >= args.runner_total:
        raise ValueError(
            f"runner index '{args.runner_idx}' must be less than available runners '{args.runner_total}'."
        )

    dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(dir, "models_op_per_op.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"File '{excel_path}' not found.")

    if args.update_model:
        update_model_file(dir, excel_path)
    else:
        process_excel_sheet(excel_path, args.runner_idx, args.runner_total)
