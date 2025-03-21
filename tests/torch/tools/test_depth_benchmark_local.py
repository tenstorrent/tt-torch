# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import yaml
import os
import xml.etree.ElementTree as ET
import ast
import pandas as pd


def parse_tests_from_matrix(yaml_file):
    """
    Parse the list of tests from the test matrix in the given YAML file.

    Args:
        yaml_file (str): Path to the YAML file.

    Returns:
        list: A single list containing all the tests from the matrix.
    """
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    # Extract the test matrix. Auto ignores comments
    matrix = data["jobs"]["tests"]["strategy"]["matrix"]["build"]

    # Combine all tests into a single list
    all_tests = []
    for entry in matrix:
        if "tests" in entry:
            all_tests.extend(entry["tests"])
    return all_tests


def test_depth_benchmark():
    yaml_file = (
        ".github/workflows/run-depth-benchmark-tests.yml"  # Path to the YAML file
    )
    all_tests = parse_tests_from_matrix(yaml_file)

    print("Combined List of Tests:")
    print(len(all_tests), "tests found in the workflow file.")

    # run a mini benchmark and evaluate that?

    # check the list of tests inside the test_data folder
    test_data_dir = "tests/torch/tools/depth_benchmark_data"  # this is bad
    found_tests = []
    result_tuples = []
    parsed_files = 0
    for file in os.listdir(test_data_dir):
        if not file.endswith(".xml"):
            continue
        parsed_files += 1
        tree = ET.parse(os.path.join(test_data_dir, file))
        root = tree.getroot()
        testcases = tree.findall(".//testcase")
        print(f"Found {len(testcases)} testcases in {file}.")
        for testcase in testcases:
            test_name = testcase.attrib["name"]
            short_name = testcase.attrib["classname"]

            found_tests.append(test_name)
            tags_property = testcase.find(".//property[@name='tags']")
            if tags_property is not None:
                testcase_tags = tags_property.attrib["value"]

                # Convert the tags string to a dictionary
                testcase_tags_dict = ast.literal_eval(testcase_tags)
                max_achieved_compile_depth = testcase_tags_dict.get(
                    "max_achieved_compile_depth"
                )
                result_tuples.append((test_name, max_achieved_compile_depth))
            else:
                assert False, f"No tags property found for test: {test_name}."

    print(len(found_tests), f"tests found in {parsed_files} fused crashsafe XML files.")

    # Compare the two lists
    missing_tests = set(all_tests) - set(found_tests)
    assert not missing_tests, f"Missing tests: {missing_tests}"

    print(
        f"Test set in workflow file matches the XML testcases found in {parsed_files} fused crashsafe XML files."
    )

    df = pd.DataFrame(
        result_tuples, columns=["Model Name", "Max Achieved Compile Depth"]
    )
    print(df.to_string())
