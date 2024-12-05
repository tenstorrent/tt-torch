# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import csv
import os
import re
import shutil
import subprocess
import sys
import xlsxwriter

from pathlib import Path


def find_ttir_files(directory):
    ttir_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mlir"):
                ttir_files.append(os.path.join(root, file))
    return ttir_files


def check_requirements():
    result_ttmlir = shutil.which("ttmlir-opt")
    if result_ttmlir is None:
        print("ttmlir-opt not found. Please install tt-mlir compiler.")
        return False

    result_ttrt = shutil.which("ttrt")
    if result_ttrt is None:
        print("ttrt not found. Please install tt-mlir compiler.")
        return False

    result_translate = shutil.which("ttmlir-translate")
    if result_translate is None:
        print("ttmlir-translate not found. Please install tt-mlir compiler.")
        return False

    return True


def create_output_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    Path(directory).mkdir(parents=True, exist_ok=True)


def generate_ttrt_artifacts(directory):
    cmd = ["ttrt", "query", "--save-artifacts", "--artifact-dir", directory]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    return result


def run_ttir_to_ttnn_pipeline(file, artifact, output_dir):
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
    file_name = os.path.basename(file)
    ttnn_file = os.path.join(output_dir, file_name)
    cmd = ["ttmlir-translate", "--ttnn-to-flatbuffer", ttnn_file]

    file_name = Path(file).stem
    output_file = open(os.path.join(output_dir, file_name + ".ttnn"), "wb")

    result = subprocess.run(cmd, stdout=output_file)


def execute_ttrt(file, output_dir):
    file_name = Path(file).stem
    fbb_file = os.path.join(output_dir, file_name + ".ttnn")
    log_fbb_file = os.path.join(output_dir, file_name + "_ttrt.log")
    fd_log = open(log_fbb_file, "w")
    cmd = ["ttrt", "run", fbb_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    fd_log.write(f"return code: {result.returncode}\n")
    fd_log.write("stdout:::\n")
    fd_log.write(result.stdout)
    fd_log.write("stderr:::\n")
    fd_log.write(result.stderr)
    return result


def parse_ttrt_output(result, filename, output_dir):
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


def execute_ttir_tests():
    if check_requirements() == False:
        return

    dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(dir, "mlir_tests", "ttir")
    output_dir = os.path.join(dir, "mlir_tests", "output", "ttir")
    ttrt_dir = os.path.join(dir, "mlir_tests", "output", "ttrt-artifacts")
    create_output_dir(output_dir)
    ttrt_output = generate_ttrt_artifacts(ttrt_dir)

    if ttrt_output.returncode != 0:
        print("Failed to generate ttrt-artifacts")
        print(ttrt_output.stderr)

    ttrt_artifacts = os.path.join(ttrt_dir, "system_desc.ttsys")
    ttir_files = find_ttir_files(input_dir)
    total_tests = len(ttir_files)
    if total_tests == 0:
        print("No ttir test file found.")
        return

    xlsx_file = os.path.join(dir, "ttrt_results.xlsx")
    if os.path.exists(xlsx_file):
        os.remove(xlsx_file)
    workbook = xlsxwriter.Workbook(xlsx_file)
    bold = workbook.add_format({"bold": True})
    worksheet = workbook.add_worksheet("ttrt_results")
    row = 0
    header = (
        "Test name",
        "Status",
        "TTIR->TTNN backend",
        "TTRT Error",
        "TTRT dump",
    )
    worksheet.write_row(row, 0, header, bold)
    row += 1

    counter = 1
    ttnn_success = 0
    ttrt_success = 0
    for file in ttir_files:
        print(
            f"Processing [{counter}/{total_tests}] :: {os.path.basename(file)}",
            file=sys.stderr,
        )
        filename = Path(file).stem
        counter += 1
        ttnn_errorcode, ttnn_error = run_ttir_to_ttnn_pipeline(
            file, ttrt_artifacts, output_dir
        )
        if ttnn_errorcode != 0:
            print("Failed: TTIR->TTNN failed", file=sys.stderr)
            row_data = [
                file,
                "TTIR->TTNN failed",
                ttnn_error,
                "",
                "",
            ]
            worksheet.write_row(row, 0, row_data)
            row += 1
            continue

        ttnn_success += 1
        generate_flatbuffer(file, output_dir)
        ttrt_result = execute_ttrt(file, output_dir)
        ttrt_errorcode, ttrt_error = parse_ttrt_output(
            ttrt_result, filename, output_dir
        )

        if ttrt_errorcode == 0:
            row_data = [
                file,
                "TTRT execution succeeded",
                "",
                "",
                ttrt_result.stdout + ttrt_result.stderr,
            ]
            worksheet.write_row(row, 0, row_data)
            row += 1
            ttrt_success += 1
            continue

        print("Failed: TTRT execution failed", file=sys.stderr)
        row_data = [
            file,
            "TTRT execution failed",
            "",
            ttrt_error,
            ttrt_result.stdout + ttrt_result.stderr,
        ]
        worksheet.write_row(row, 0, row_data)
        row += 1

    print("Summary")
    print(f"Total tests: {total_tests}")
    print(f"TTIR to TTNN pipeline success {ttnn_success}/{total_tests}")
    print(f"TTRT success {ttrt_success}/{ttnn_success}")
    workbook.close()


if __name__ == "__main__":
    execute_ttir_tests()
