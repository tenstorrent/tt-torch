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


def run_iv_tests_and_generate_summary(yaml_files, summary_file="iv_test_summary.txt"):
    """
    Extract test names from YAML files, run them, and generate a summary file.

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
            "./.github/workflows/run-full-model-execution-tests-nightly.yml",
        ]
        run_iv_tests_and_generate_summary(yaml_files)
