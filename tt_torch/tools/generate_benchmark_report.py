# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import xml.etree.ElementTree as ET
import yaml
import ast
import xlsxwriter
from tt_torch.tools.crashsafe_utils import achieved_depth_mapping
from datetime import datetime


def parse_benchmark_xml(xml_dir):
    """
    Parse XML files in the given directory to extract test case details.

    Args:
        xml_dir (str): Path to the directory containing XML files.

    Returns:
        list: A list of tuples containing (testcase_name, execution_time, max_achieved_compile_depth).
    """
    result_tuples = []
    for file in os.listdir(xml_dir):
        if not file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(xml_dir, file))
        root = tree.getroot()
        testcases = root.findall(".//testcase")
        testcases_in_file = 0
        for testcase in testcases:
            testcases_in_file += 1
            testcase_name = testcase.attrib.get("name")
            execution_time = 0

            try:
                start_time = testcase.find(".//property[@name='start_timestamp']")
                end_time = testcase.find(".//property[@name='end_timestamp']")
                start_dt = datetime.fromisoformat(
                    start_time.attrib["value"].replace("Z", "+00:00")
                )
                end_dt = datetime.fromisoformat(
                    end_time.attrib["value"].replace("Z", "+00:00")
                )
                execution_time = (end_dt - start_dt).total_seconds()
            except Exception as e:
                print(
                    f"Error parsing execution time for testcase {testcase_name}. This is okay and means the pytest was abruptly killed without being able to flush the end_timestamp, but will result in a 0 execution time recorded.",
                    e,
                )

            tags_property = testcase.find(".//property[@name='tags']")
            max_achieved_compile_depth = None
            if tags_property is not None:
                tags = tags_property.attrib.get("value", "{}")
                tags_dict = ast.literal_eval(tags)
                max_achieved_compile_depth = tags_dict.get("max_achieved_compile_depth")
            result_tuples.append(
                (testcase_name, execution_time, max_achieved_compile_depth)
            )
        print(f"Found {testcases_in_file} testcases in {file}.")
    print(f"Found total {len(result_tuples)} testcases in {xml_dir}.")
    return result_tuples


def parse_tests_from_matrix(yaml_file):
    """
    Parse the list of tests from the test matrix in the given YAML file.

    Args:
        yaml_file (str): Path to the YAML file.

    Returns:
        list: A list containing all the tests from the matrix.
    """
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)
    matrix = data["jobs"]["tests"]["strategy"]["matrix"]["build"]
    all_tests = []
    for entry in matrix:
        if "tests" in entry:
            test_entry = entry["tests"]
            if isinstance(test_entry, list):
                all_tests.extend(test_entry)
            elif isinstance(test_entry, str):
                all_tests.extend(test_entry.split())
            else:
                assert False, f"Invalid tests format specification from {file}"
    print(f"Found {len(all_tests)} tests in {yaml_file}.")
    return all_tests


def generate_spreadsheet(output_file, benchmark_data, compile_tests, execution_tests):
    """
    Generate a spreadsheet with benchmark data and test coverage information.

    Args:
        output_file (str): Path to the output spreadsheet file.
        benchmark_data (list): List of tuples containing benchmark data.
        compile_tests (list): List of tests run in compile test cases.
        execution_tests (list): List of tests run in execution test cases.
    """
    # Define depth hierarchy for comparison
    depth_hierarchy = achieved_depth_mapping

    workbook = xlsxwriter.Workbook(output_file)
    worksheet = workbook.add_worksheet("Benchmark Report")

    # Write headers
    headers = [
        "Testcase Name",
        "Execution Time (s)",
        "Max Achieved Compile Depth",
        "In Compile Testcases",
        "In Execution Testcases",
        "Regressed?",
        "Promotable?",
    ]
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)

    color_formats = {
        "green": workbook.add_format({"bg_color": "#4CAF50"}),
        "red": workbook.add_format({"bg_color": "#F44336"}),
    }

    # Initialize counters for regressed and promotable rows
    regressed_count = 0
    promotable_count = 0

    # Write data
    for row, (testcase_name, execution_time, max_depth) in enumerate(
        benchmark_data, start=1
    ):
        in_compile = testcase_name in compile_tests
        in_execution = testcase_name in execution_tests

        # Determine regression
        regressed = (
            in_execution and depth_hierarchy[max_depth] < depth_hierarchy["EXECUTE"]
        ) or (in_compile and depth_hierarchy[max_depth] < depth_hierarchy["TTNN_IR"])
        if regressed:
            regressed_count += 1

        # Determine promotion
        promotable = (
            not in_execution
            and depth_hierarchy[max_depth] >= depth_hierarchy["EXECUTE"]
        ) or (
            not in_compile
            and not in_execution
            and depth_hierarchy[max_depth] >= depth_hierarchy["TTNN_IR"]
        )
        if promotable:
            promotable_count += 1

        # Write row data
        worksheet.write(row, 0, testcase_name)
        worksheet.write(row, 1, execution_time)
        worksheet.write(row, 2, max_depth)
        worksheet.write(row, 3, "Yes" if in_compile else "No")
        worksheet.write(row, 4, "Yes" if in_execution else "No")
        worksheet.write(row, 5, "Yes" if regressed else "No")
        worksheet.write(row, 6, "Yes" if promotable else "No")

    # Apply conditional formatting for "Regressed?" column (column 5)
    worksheet.conditional_format(
        1,
        5,
        len(benchmark_data),
        5,  # From row 1 to the last row in column 5
        {
            "type": "text",
            "criteria": "containing",
            "value": "Yes",
            "format": color_formats["red"],
        },
    )

    # Apply conditional formatting for "Promotable?" column (column 6)
    worksheet.conditional_format(
        1,
        6,
        len(benchmark_data),
        6,  # From row 1 to the last row in column 6
        {
            "type": "text",
            "criteria": "containing",
            "value": "Yes",
            "format": color_formats["green"],
        },
    )

    # Print counts of regressed and promotable rows
    print(f"Number of regressed rows: {regressed_count}")
    print(f"Number of promotable rows: {promotable_count}")

    print(f"Wrote {len(benchmark_data)} rows to {output_file}.")

    workbook.close()


def main(xml_dir, compile_yaml, execution_yaml, execution_nightly_yaml, output_file):
    """
    Main function to generate the benchmark report.

    Args:
        xml_dir (str): Path to the directory containing benchmark XML files.
        compile_yaml (str): Path to the YAML file for compile test cases.
        execution_yaml (str): Path to the YAML file for execution test cases.
        execution_nightly_yaml (str): Path to the YAML file for nightly execution test cases.
        output_file (str): Path to the output spreadsheet file.
    """

    print("Parsing benchmark XML files...")
    benchmark_data = parse_benchmark_xml(xml_dir)

    print("Parsing compile test cases from YAML...")
    compile_tests = parse_tests_from_matrix(compile_yaml)

    print("Parsing execution test cases from YAML...")
    execution_tests = parse_tests_from_matrix(execution_yaml)
    execution_tests += parse_tests_from_matrix(execution_nightly_yaml)

    generate_spreadsheet(output_file, benchmark_data, compile_tests, execution_tests)
    print(f"Benchmark report generated: {output_file}")


if __name__ == "__main__":
    """
    Example usage:

    python generate_benchmark_report.py \
    --xml-dir tests/torch/tools/depth_benchmark_data \
    --compile-yaml .github/workflows/run-e2e-compile-tests.yml \
    --execution-yaml .github/workflows/run-full-model-execution-tests.yml \
    --execution-nightly-yaml .github/workflows/run-full-model-execution-tests-nightly.yml \
    --output-file benchmark_report.xlsx
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate Benchmark Report")
    parser.add_argument(
        "--xml-dir",
        required=True,
        help="Path to the directory containing benchmark XML files",
    )
    parser.add_argument(
        "--compile-yaml",
        required=True,
        help="Path to the YAML file for compile test cases",
    )
    parser.add_argument(
        "--execution-yaml",
        required=True,
        help="Path to the YAML file for execution test cases",
    )
    parser.add_argument(
        "--execution-nightly-yaml",
        required=True,
        help="Path to the YAML file for nightly execution test cases",
    )
    parser.add_argument(
        "--output-file", required=True, help="Path to the output spreadsheet file"
    )

    args = parser.parse_args()
    main(
        args.xml_dir,
        args.compile_yaml,
        args.execution_yaml,
        args.execution_nightly_yaml,
        args.output_file,
    )
