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
import subprocess


def scan_workflow_test_matrices():
    """
    Scans all .yml files in .github/workflows, validates test matrices, and asserts:
    - No duplicates within a single test matrix.
    - The set of tests in any YAML file is a subset of the set of all tests.
    """

    workflows_dir = ".github/workflows"

    # iterate through all pytest files in tests/models and find all tests/parameterizations
    # sudo apt install -y libgl1 libglx-mesa0 (may be needed locally to parse yolov4 test)
    all_tests = set(enumerate_all_tests(filter_full_eval=False, test_dir="tests"))
    all_tests.update(
        [test.split("[")[0] for test in all_tests]
    )  # strip off parameterization to allow unparameterized tests in ymls

    print(f"Found a total of {len(all_tests)} tests in tests.")
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

                    # We may decide to specify a group of tests by file, directory, test module or something other than full parameterization
                    # In these cases, reuse pytest collect-only on the test string literal and see if it resolves to any tests
                    unknown_tests_to_remove = []
                    for unparameterized_test in unknown_tests:
                        print(
                            f'Searching for child tests for non-fully parameterized test "{unparameterized_test}":'
                        )

                        child_tests = enumerate_all_tests(
                            filter_full_eval=False,
                            test_dir=unparameterized_test,
                            dry_run=False,
                        )

                        if child_tests:
                            print(
                                f'Resolved {len(child_tests)} subtests for "{unparameterized_test}". This is a real test.'
                            )
                            unknown_tests_to_remove.append(unparameterized_test)
                        else:
                            print(
                                f'Could not find any child tests for "{unparameterized_test}". This is likely an invalid test.'
                            )

                    # Remove the unparameterized tests that resolved to real tests
                    for unparameterized_test in unknown_tests_to_remove:
                        unknown_tests.remove(unparameterized_test)

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


def dissect_runtime_verification_report(log_folder, output_xlsx):
    """
    Parses runtime intermediate verification report log files in a folder
    and generates an Excel file with multiple worksheets.

    Args:
        log_folder (str): Path to the folder containing log files.
        output_xlsx (str): Path to the output Excel file.
    """

    pytest_name_pattern = r"(tests/models/.+\.py::[^\s]+)"
    model_name_pattern = r"\[MODEL NAME\]\s+(.+)"
    final_row_pattern = r"Final Row:\s*(.+)"
    first_failing_op_pattern = r"First Failing Op with PCC < [^:]+:\s*(.+)"

    # Create an Excel file
    workbook = xlsxwriter.Workbook(output_xlsx)

    # Define formats
    formats = {
        "default": workbook.add_format({"border": 1}),
        "red": workbook.add_format({"bg_color": "#FFCCCC", "border": 1}),
        "yellow": workbook.add_format({"bg_color": "#FFF2CC", "border": 1}),
        "green": workbook.add_format({"bg_color": "#C6EFCE", "border": 1}),
        "bold": workbook.add_format({"bold": True, "border": 1}),
    }

    model_names = []
    summary = []
    corrupt_logs = []
    final_rows = []
    first_failing_ops = []

    # Helper function to parse numeric values or handle errors
    def parse_numeric(value):
        try:
            ret = float(value.strip("[]"))
            if ret != ret or ret == float("inf") or ret == -float("inf"):
                return None
            return ret
        except (ValueError, AttributeError):
            return None  # Treat non-numeric values (e.g., "ERROR") as None

    # Iterate through all log files in the folder
    for log_file in os.listdir(log_folder):
        print(f"Processing file {log_file}")
        log_path = os.path.join(log_folder, log_file)

        if not os.path.isfile(log_path):
            print(f"Skipping {log_file} as it is not a file.")
            continue

        rows = []
        csv_data = []
        pytest_full_name = None
        model_name = None
        inside_csv = False
        final_row = None
        first_failing_op = None
        first_failing_op_row = None

        # Read the log file
        with open(log_path, "r") as file:
            for line in file:
                # Check for start and end markers
                if "[Start Intermediate Verification Report]" in line:
                    inside_csv = True
                    continue
                if "[End Intermediate Verification Report]" in line:
                    inside_csv = False
                    continue

                # Extract pytest full test name
                if not pytest_full_name:
                    pytest_match = re.search(pytest_name_pattern, line)
                    if pytest_match:
                        pytest_full_name = pytest_match.group(1)

                # Extract model name
                if not model_name:
                    model_match = re.search(model_name_pattern, line)
                    if model_match:
                        model_name = model_match.group(1)
                        model_name = re.sub(r"[\s\\/:*?\"<>|\[\]\(\)]", "_", model_name)
                        print(f"\tFound data for model {model_name}")

                # Collect CSV data
                if inside_csv:
                    csv_data.append(line.strip())

                # Extract Final Row
                final_row_match = re.search(final_row_pattern, line)
                if final_row_match:
                    final_row = final_row_match.group(1).strip()

                # Extract First Failing Op
                first_failing_op_match = re.search(first_failing_op_pattern, line)
                if first_failing_op_match:
                    first_failing_op_details = (
                        first_failing_op_match.group(1).strip().split(",")
                    )
                    first_failing_op = first_failing_op_details[0]
                    first_failing_op_row = first_failing_op_details

        if final_row:
            final_rows.append((log_file, final_row))
        if first_failing_op:
            first_failing_ops.append((log_file, first_failing_op, first_failing_op_row))
        # Check for missing markers
        if not csv_data:
            if pytest_full_name:
                corrupt_logs.append(pytest_full_name)
            continue

        # Skip if no model name was found
        if not model_name:
            print(f"Could not extract model name from {log_file}. Skipping...")
            continue

        # Parse CSV data
        header = csv_data[0].split(",")
        for row in csv_data[1:]:
            values = row.split(",")
            if len(values) != len(header):
                continue
            row_data = dict(zip(header, values))
            rows.append(
                [
                    row_data.get("NodeName"),
                    parse_numeric(row_data.get("PCC")),
                    parse_numeric(row_data.get("ATOL")),
                    row_data.get("ErrorMessage"),
                    parse_numeric(row_data.get("FlattenedPCC")),
                    parse_numeric(row_data.get("FlattenedATOL")),
                    row_data.get("FlattenedErrorMessage"),
                ]
            )

        # Handle duplicate model names + 31 char limit
        model_name = model_name.lower()
        original_model_name = model_name
        if model_name in model_names:
            model_name = (
                f"{len([m for m in model_names if m == model_name])}_{model_name}"
            )
        model_names.append(original_model_name)

        # Create a worksheet for the module
        worksheet = workbook.add_worksheet(model_name[:30])

        # Define header and write it
        headers = [
            "Node Name",
            "PCC",
            "ATOL",
            "Error Message",
            "Flattened PCC",
            "Flattened ATOL",
            "Flattened Error Message",
        ]
        for col_num, header in enumerate(headers):
            worksheet.write(0, col_num, header)

        # Adjust column widths
        column_widths = [20, 10, 10, 50, 15, 15, 50]  # Define widths for each column
        for col_num, width in enumerate(column_widths):
            worksheet.set_column(col_num, col_num, width)

        # Write data rows with conditional formatting
        for row_num, row in enumerate(rows, start=1):
            for col_num, cell in enumerate(row):
                if first_failing_op and row[0] == first_failing_op:
                    worksheet.write(row_num, col_num, cell, formats["bold"])
                    worksheet.write_comment(row_num, col_num, "First failing op")
                elif col_num == 3 or col_num == 6:  # Error message columns
                    if cell:
                        worksheet.write(row_num, col_num, cell, formats["red"])
                    else:
                        worksheet.write(row_num, col_num, cell, formats["default"])
                elif col_num == 1 or col_num == 4:  # PCC and Flattened PCC columns
                    if cell is not None:
                        if cell < 0.95:
                            worksheet.write(row_num, col_num, cell, formats["red"])
                        elif 0.95 <= cell < 0.99:
                            worksheet.write(row_num, col_num, cell, formats["yellow"])
                        else:
                            worksheet.write(row_num, col_num, cell, formats["green"])
                    else:
                        worksheet.write(row_num, col_num, cell, formats["default"])
                else:
                    worksheet.write(row_num, col_num, cell, formats["default"])

    # Add a summary worksheet
    summary_sheet = workbook.add_worksheet("Summary")
    summary_sheet.write(0, 0, "Log File")
    summary_sheet.write(0, 1, "Final Row")
    summary_sheet.write(0, 2, "First Failing Op")
    summary_sheet.write(0, 3, "First Failing Op Row")
    summary_sheet_column_widths = [20, 50, 50, 100]  # Define widths for each column
    for col_num, width in enumerate(summary_sheet_column_widths):
        summary_sheet.set_column(col_num, col_num, width)

    for idx, (log_file, final_row) in enumerate(final_rows, start=1):
        summary_sheet.write(idx, 0, log_file)
        summary_sheet.write(idx, 1, final_row)

        # Find the corresponding first failing op
        failing_op = next((op for op in first_failing_ops if op[0] == log_file), None)
        if failing_op:
            summary_sheet.write(idx, 2, failing_op[1])
            summary_sheet.write(idx, 3, str(failing_op[2]))

    # Close the workbook
    workbook.close()
    print(f"Verification report saved to {output_xlsx}")

    # Print corrupt logs
    if corrupt_logs:
        print("\nCorrupt Logs:")
        for log in corrupt_logs:
            print(log)
    else:
        print("\nNo corrupt logs found.")


def run_iv_tests_and_generate_summary(yaml_files, summary_file="iv_test_summary.txt"):
    """
    Extract test names from YAML files, run them with intermediate verification report, and generate a summary file.
    Mostly a validation utility for runtime intermediate task.
    Args:
        yaml_files (list): List of paths to YAML files containing test matrices.
        summary_file (str): Path to the summary file to write test results.
    """
    all_tests = []

    # Extract test names from each YAML file
    for yaml_file in yaml_files:
        print(f"Parsing tests from {yaml_file}...")
        tests = parse_tests_from_matrix(yaml_file)
        all_tests.extend(tests)

    print(f"Found {len(all_tests)} tests in total.")

    # all_tests = all_tests[::2]

    # Prepare the summary file
    with open(summary_file, "w") as summary:
        summary.write("Test Name | Result\n")
        summary.write("------------------\n")

        # Run each test and log the result
        for test in all_tests:
            test_name = test.strip()
            log_file = f"{test_name.replace('/', '_').replace(':', '_')}.ri.log"
            print(f"Running test: {test_name}")
            command = f"TT_TORCH_VERIFY_INTERMEDIATES=1 pytest -svv {test_name} | tee {log_file}"

            # Run the test and capture the return code
            result = subprocess.run(command, shell=True)
            test_result = "PASSED" if result.returncode == 0 else "FAILED"

            # Write the result to the summary file
            summary.write(f"{test_name} | {test_result}\n")

    print(f"Test summary written to {summary_file}")


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
        metavar=("LOG_FOLDER", "OUTPUT_XLSX"),
        help="Dissect runtime intermediate verification reports in a folder and save them as an Excel file.",
    )
    parser.add_argument(
        "--run-iv-tests",
        action="store_true",
        help="Run full eval tests with intermediate verification from the given YAML files.",
    )

    args = parser.parse_args()

    if args.scan_workflows:
        scan_workflow_test_matrices()

    if args.run_iv_tests:
        yaml_files = [
            "./.github/workflows/run-full-model-execution-tests.yml",
            # "./.github/workflows/run-full-model-execution-tests-nightly.yml",
        ]
        run_iv_tests_and_generate_summary(yaml_files)

    if args.dissect_report:
        log_folder, output_xlsx = args.dissect_report
        dissect_runtime_verification_report(log_folder, output_xlsx)
