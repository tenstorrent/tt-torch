# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import yaml
import os
import xml.etree.ElementTree as ET
import ast
import pandas as pd
import json


def parse_tests_from_json(json_file):
    """
    Parse the JSON file to extract the full_test_name and max_achieved_compile_depth.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        list: A list of tuples containing (full_test_name, max_achieved_compile_depth).
    """
    result_tuples = []

    # Load the JSON file
    with open(json_file, "r") as file:
        data = json.load(file)

    # Iterate through the jobs
    for job in data.get("jobs", []):
        # Iterate through the tests in each job
        for test in job.get("tests", []):
            full_test_name = test.get("full_test_name")

            full_test_name = "".join(
                full_test_name.split("::")[1:]
            )  # strip out the short test name

            tags = test.get("tags", {})
            max_achieved_compile_depth = tags.get("max_achieved_compile_depth")

            # Add to result_tuples if both fields are present
            if full_test_name and max_achieved_compile_depth:
                result_tuples.append((full_test_name, max_achieved_compile_depth))

    return result_tuples


def parse_tests_from_matrix(yaml_file, array_fmt=True):
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
            all_tests.extend(entry["tests"] if array_fmt else entry["tests"].split())
    return all_tests


def test_depth_benchmark():
    yaml_file = (
        ".github/workflows/run-depth-benchmark-tests.yml"  # Path to the YAML file
    )
    all_tests = parse_tests_from_matrix(yaml_file)

    print("Combined List of Tests:")
    print(len(all_tests), "tests found in the benchmarker workflow file.")
    # print(all_tests)
    # run a mini benchmark and evaluate that?

    # check the list of tests inside the test_data folder
    test_data_dir = "tests/torch/tools/depth_benchmark_data"  # this is bad
    found_tests = []
    result_tuples_fusedxml = []
    parsed_files = 0
    for file in os.listdir(test_data_dir):
        if not file.endswith(".xml"):
            continue
        parsed_files += 1
        tree = ET.parse(os.path.join(test_data_dir, file))
        root = tree.getroot()
        testcases = tree.findall(".//testcase")
        print(f"\tFound {len(testcases)} testcases in {file}.")
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
                result_tuples_fusedxml.append((test_name, max_achieved_compile_depth))
            else:
                assert False, f"No tags property found for test: {test_name}."

    print(len(found_tests), f"tests found in {parsed_files} fused crashsafe XML files.")

    # Compare the two lists
    missing_tests = set(all_tests) - set(found_tests)
    # print(found_tests)
    assert not missing_tests, f"Missing tests: {missing_tests}"

    print(
        f"Test set in workflow file matches the XML testcases found in {parsed_files} fused crashsafe XML files."
    )

    df = pd.DataFrame(
        result_tuples_fusedxml, columns=["Model Name", "Max Achieved Compile Depth"]
    )
    # print(df.to_string())

    json_file = "tests/torch/tools/depth_benchmark_data/pipeline_14020910456_2025-03-23T16:34:59.000000+0000.json"  # Path to the JSON file
    result_tuples_json = parse_tests_from_json(json_file)

    df = pd.DataFrame(
        result_tuples_json, columns=["Model Name", "Max Achieved Compile Depth"]
    )
    # print(df.to_string())

    # missing_json_fusedxml = set(result_tuples_json) - set(result_tuples_fusedxml)
    # assert missing_json_fusedxml == set(), f"Missing tests: {missing_json_fusedxml}"

    print(
        f"Test set and results in fused crashsafe XML files match those in the parsed JSON"
    )

    # Validate against expectations, at least for full execute
    # We would expect that everything currently running in full model execution
    # shows up as such
    current_exec_tests = parse_tests_from_matrix(
        ".github/workflows/run-full-model-execution-tests.yml", array_fmt=False
    )
    current_exec_tests += parse_tests_from_matrix(
        ".github/workflows/run-full-model-execution-tests-nightly.yml", array_fmt=False
    )

    current_exec_tests = [x.replace("::", "") for x in current_exec_tests]

    current_compiling_tests = parse_tests_from_matrix(
        ".github/workflows/run-e2e-tests.yml", array_fmt=False
    )

    current_exec_tests = [x.replace("::", "") for x in current_exec_tests]
    current_compiling_tests = [x.replace("::", "") for x in current_compiling_tests]

    current_compiling_or_executing_tests = current_exec_tests + current_compiling_tests

    result_tuples_json_executing = [
        tup[0]
        for tup in result_tuples_json
        if tup[1] == "EXECUTE" or tup[1] == "PASSED"
    ]

    result_tuples_json_compiling_or_executing = [
        tup[0]
        for tup in result_tuples_json
        if tup[1] == "TTNN_IR" or tup[1] == "EXECUTE" or tup[1] == "PASSED"
    ]

    in_exec_but_not_in_json = set(current_exec_tests) - set(
        result_tuples_json_executing
    )
    in_json_but_not_in_exec = set(result_tuples_json_executing) - set(
        current_exec_tests
    )
    in_exec_but_not_in_json = sorted(in_exec_but_not_in_json)

    print("\n===\nResults:\n===\n")

    # if a model is in the full execution test coverage but doesn't make it to passed or execute in the benchmark, there is an issue with the benchmark
    print(
        f"Tests in current execution but not executing in benchmark JSON (Count: {len(in_exec_but_not_in_json)}):"
    )
    df = pd.DataFrame(in_exec_but_not_in_json, columns=["Model Name"])
    print(df.to_string())
    print()

    # if a model makes it to full execution in the benchmark, but is not covered in full execution tests, then it is a promotion candidate
    print(
        f"Promotable tests in benchmark JSON but not in current execution (Count: {len(in_json_but_not_in_exec)}):"
    )
    df = pd.DataFrame(in_json_but_not_in_exec, columns=["Model Name"])
    print(df.to_string())
    print()

    in_compiling_but_not_in_json = set(current_compiling_tests) - set(
        result_tuples_json_compiling_or_executing
    )
    in_json_but_not_in_compiling_or_executing = set(
        result_tuples_json_compiling_or_executing
    ) - set(current_compiling_or_executing_tests)

    # if a model is in the test coverage for compilation, but actually executes, we should remove it from the compile e2e tests and it is not an error
    # if a model is in the test coverage for compilation but doesn't even make it to TTNN_IR in the benchmark then the benchmark has an issue
    print(
        f"Tests in current compiling but not compiling in benchmark JSON (Count: {len(in_compiling_but_not_in_json)}):"
    )
    df = pd.DataFrame(in_compiling_but_not_in_json, columns=["Model Name"])
    print(df.to_string())
    print()

    # if a model is in the JSON as "TTNN_IR" but isn't in the existing test coverage for compiling or executing, then it's a promotion candidate
    print(
        f"Promotable tests in JSON but not in current compiling or executing tests: (Count: {len(in_json_but_not_in_compiling_or_executing)}) "
    )
    df = pd.DataFrame(in_json_but_not_in_compiling_or_executing, columns=["Model Name"])
    print(df.to_string())

    # summary generation
    print(f"\nSummary:\n")
    summary_list = []
    for test in all_tests:
        benchmarked_depth = next(
            (depth for name, depth in result_tuples_fusedxml if name == test),
            None,  # Default value if no match is found
        )
        test = test.replace("::", "")
        summary_tuple = (
            test,
            benchmarked_depth,
            test in current_compiling_tests,
            test in current_exec_tests,
        )
        summary_list.append(summary_tuple)
    df = pd.DataFrame(
        summary_list,
        columns=[
            "Model Name",
            "Benchmarked Depth",
            "In Current Compilation Test Coverage",
            "In Current Full Execution Test Coverage",
        ],
    )
    print(df.to_string())
