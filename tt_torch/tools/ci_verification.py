# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import yaml
import pprint
import argparse
from tt_torch.tools.generate_benchmark_report import parse_tests_from_matrix
from tt_torch.tools.benchmark_promotion import enumerate_all_tests
import difflib
import re
import xlsxwriter


def scan_workflow_test_matrices():
    """
    Scans all .yml files in .github/workflows, validates test matrices, and asserts:
    - No duplicates within a single test matrix.
    - The set of tests in any YAML file is a subset of the set of all tests.
    """

    workflows_dir = ".github/workflows"

    # iterate through all pytest files in tests/models and find all tests/parameterizations
    # sudo apt install -y libgl1 libglx-mesa0 (may be needed locally to parse yolov4 test)
    all_tests = set(enumerate_all_tests(filter_full_eval=False))
    all_tests.update(
        [test.split("[")[0] for test in all_tests]
    )  # strip off parameterization to allow unparameterized tests in ymls

    print(f"Found a total of {len(all_tests)} tests in tests/models.")
    pp = pprint.PrettyPrinter(indent=2)
    workflows_are_valid = True

    for filename in os.listdir(workflows_dir):
        if filename.endswith(".yml") or filename.endswith(".yaml"):
            filepath = os.path.join(workflows_dir, filename)
            with open(filepath, "r") as file:
                content = yaml.safe_load(file)
                if "matrix" in content.get("jobs", {}).get("tests", {}).get(
                    "strategy", {}
                ):
                    matrix_content = (
                        content.get("jobs", {})
                        .get("tests", {})
                        .get("strategy", {})
                        .get("matrix")
                    )
                    build_content = matrix_content.get("build", {})
                    if build_content == {}:
                        print(
                            f"No build matrix found in {filename}. It is probably dynamically generated - the contents of the matrix are \
                            {build_content}"
                        )
                        continue

                    tests = parse_tests_from_matrix(filepath)

                    # Assert no duplicates within the test matrix
                    duplicates = [test for test in tests if tests.count(test) > 1]
                    if duplicates:
                        print(f"{len(duplicates)} duplicate tests in {filename}:")
                        pp.pprint(duplicates)
                        workflows_are_valid = False
                    # Assert all tests are a subset of the known tests
                    unknown_tests = [test for test in tests if test not in all_tests]
                    if unknown_tests:
                        print(f"{len(unknown_tests)} nonexistent tests in {filename}:")
                        for i, unk in enumerate(unknown_tests):
                            # Use difflib to show closest matches in the known tests
                            closest_match = difflib.get_close_matches(
                                unk, list(all_tests), n=1, cutoff=0.6
                            )
                            print(f"  {i}. {unk}\n\t(Did you mean {closest_match}?)")

                        workflows_are_valid = False

    assert (
        workflows_are_valid
    ), "There are some errors in the workflow test matrices. Please fix the duplicates or nonexistent tests."
    print("No issues found in the workflow test matrices. All matrices are valid!")


def dissect_runtime_verification_report(log_file, output_xlsx):
    """
    Parses a runtime intermediate verification report log file and generates an Excel file.

    Args:
        log_file (str): Path to the log file containing the verification report.
        output_xlsx (str): Path to the output Excel file.
    """
    # Regex patterns to extract data
    error_pattern = r"Metrics for (\w+): ERROR: (.+)"
    metrics_pattern = r"Metrics for (\w+): pcc \[([\d.]+)\]\tatol \[([\d.]+)\]"

    # Data storage
    rows = []

    # Read the log file
    with open(log_file, "r") as file:
        for line in file:
            # Match error lines
            error_match = re.match(error_pattern, line)
            if error_match:
                node_name, error_message = error_match.groups()
                rows.append([node_name, None, None, error_message])
                continue

            # Match metrics lines
            metrics_match = re.match(metrics_pattern, line)
            if metrics_match:
                node_name, pcc, atol = metrics_match.groups()
                rows.append([node_name, float(pcc), float(atol), None])

    # Create an Excel file
    workbook = xlsxwriter.Workbook(output_xlsx)
    worksheet = workbook.add_worksheet("Verification Report")

    # Define header and write it
    headers = ["Node Name", "PCC", "ATOL", "Error Message"]
    for col_num, header in enumerate(headers):
        worksheet.write(0, col_num, header)

    # Define formats
    default_format = workbook.add_format({"border": 1})
    red_format = workbook.add_format({"bg_color": "#FFCCCC", "border": 1})

    # Write data rows
    for row_num, row in enumerate(rows, start=1):
        for col_num, cell in enumerate(row):
            # Apply conditional formatting directly while writing
            if (row[1] is not None and row[1] < 0.99) or (row[3] is not None):
                worksheet.write(row_num, col_num, cell, red_format)
            else:
                worksheet.write(row_num, col_num, cell, default_format)

    # Close the workbook
    workbook.close()
    print(f"Verification report saved to {output_xlsx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility script for CI reflection.")
    parser.add_argument(
        "--scan-workflows",
        action="store_true",
        help="Scan .github/workflows for test matrices and validate them.",
    )
    parser.add_argument(
        "--dissect-report",
        nargs=2,
        metavar=("LOG_FILE", "OUTPUT_XLSX"),
        help="Dissect a runtime intermediate verification report and save it as an Excel file.",
    )

    args = parser.parse_args()

    if args.scan_workflows:
        scan_workflow_test_matrices()

    if args.dissect_report:
        log_file, output_xlsx = args.dissect_report
        dissect_runtime_verification_report(log_file, output_xlsx)
