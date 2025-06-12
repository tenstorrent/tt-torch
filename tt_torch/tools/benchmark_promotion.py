# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re
import subprocess
from tt_torch.tools.generate_benchmark_report import parse_tests_from_matrix
from collections import Counter
import argparse
import pandas as pd
import os
import json

MAXIMUM_JOB_TIMEOUT_MINUTES = 500  # 500 minutes maximum per-job timeout
DEFAULT_JOB_TIMEOUT_MINUTES = 120


def enumerate_all_tests(filter_full_eval=True, test_dir="tests/models", dry_run=False):
    print(f"Running pytest collect command: pytest {test_dir} --collect-only -q")
    try:
        # Run pytest with --collect-only and capture the output
        # sudo apt install -y libgl1 libglx-mesa0 # (may be needed locally)
        if dry_run:
            return []
        result = subprocess.run(
            ["pytest", test_dir, "--collect-only", "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Check if pytest ran successfully
        if result.returncode != 0:
            raise RuntimeError(
                f"pytest failed with\n\tstderr:\n{result.stderr}\n\tstdout:\n{result.stdout}"
            )

        # Extract test names using a regex
        test_cases = re.findall(r"(\S+::\S+)", result.stdout)

        return (
            [tc for tc in test_cases if "full" in tc and "eval" in tc]
            if filter_full_eval
            else test_cases
        )

    except Exception as e:
        print(f"Error: {e}")
        return []


def generate_test_matrix(test_list):
    full_eval_test_list = test_list
    matrix = []
    n_groups = 8  # arbitrary number of groups ~ # CI test runners
    group_size = max(
        1, len(full_eval_test_list) // n_groups
    )  # Ensure group_size is at least 1

    for i in range(0, len(full_eval_test_list), group_size):
        # Slice up to the end of the list to avoid out-of-bounds errors
        group = full_eval_test_list[i : min(i + group_size, len(full_eval_test_list))]
        matrix.append(
            {
                "runs-on": "wormhole_b0",
                "name": f"benchmark_{len(matrix) + 1}",
                "tests": group,
            }
        )

    print(matrix)
    return matrix


def find_duplicates(string_array):
    counts = Counter(string_array)
    return [item for item, count in counts.items() if count > 1]


def discover_tests():
    in_tree_tests = enumerate_all_tests()
    benchmark_tests = parse_tests_from_matrix(
        ".github/workflows/run-depth-benchmark-tests.yml"
    )

    print("Verifying test sets contain no duplicates.")
    assert (
        find_duplicates(benchmark_tests) == []
    ), "There are duplicate test cases in the benchmark tests."
    assert (
        find_duplicates(in_tree_tests) == []
    ), "There are duplicate test cases in the in-tree tests."

    removed_tests = set(benchmark_tests) - set(in_tree_tests)
    added_tests = set(in_tree_tests) - set(benchmark_tests)

    if removed_tests != set():
        print(f"Warning {len(removed_tests)} pytests have been removed from tree.")
    else:
        print("No pytests have been removed from the benchmark tests.")

    # Added tests should be quarantined into their own runners and run in isolation
    # so unexpected/unrecoverable breakage doesn't cause other benchmarks to fail
    if (added_tests) != set():
        print(
            f"{len(added_tests)} pytests have been added to tree and are not in the current benchmark."
        )
        for i, t in enumerate(added_tests):
            print(f"{i+1}\t{t}")

    return added_tests


def load_balance_tests_greedy(
    test_durations, n_partitions=10, print_summary=True, do_linearize=False
):
    """
    Load balances test names into N partitions based on test execution time.

    Args:
        test_durations (dict): A dictionary where keys are test names and values are their execution times.
        n_partitions (int): Number of partitions to split the test names into.
        do_linearize (bool): If True, do not load balance and run all tests in individual partitions.

    Returns:
        list of lists: A list containing N partitions, each with a subset of test names.
    """
    # Separate tests with known durations and unknown durations (-1)
    known_tests = [
        (test_name, duration)
        for test_name, duration in test_durations.items()
        if duration != -1
    ]
    unknown_tests = [
        test_name for test_name, duration in test_durations.items() if duration == -1
    ]

    # Sort known tests by execution time in descending order
    known_tests.sort(key=lambda x: x[1], reverse=True)

    partitions = None

    if not do_linearize:
        # Initialize partitions and their total times
        partitions = [[] for _ in range(n_partitions)]
        partition_times = [0] * n_partitions

        # Distribute known tests greedily to minimize the maximum partition time
        for test_name, test_time in known_tests:
            # Find the partition with the smallest total time
            min_index = partition_times.index(min(partition_times))
            partitions[min_index].append(
                {"full-test-name": test_name, "test-duration": test_time}
            )
            partition_times[min_index] += test_time

        # Add unknown tests (-1 duration) to their own partitions
        for test_name in unknown_tests:

            partitions.append([{"full-test-name": test_name, "test-duration": -1}])
    else:
        # don't load balance and run all tests in a individual partitions
        partitions = []
        for test_name in unknown_tests:
            partitions.append([{"full-test-name": test_name, "test-duration": -1}])
        for test_name, test_time in known_tests:
            partitions.append(
                [{"full-test-name": test_name, "test-duration": test_time}]
            )

    if print_summary:
        print("\nPartition Summary:")
        for i, partition in enumerate(partitions):
            for test in partition:
                print("test in partition: ", test)
                print("\ttime ", test["test-duration"])

            partition_duration = sum(test.get("test-duration") for test in partition)
            print(f"Partition {i + 1}:")
            print(f"  Tests: {partition}")
            print(f"  Estimated Duration: {partition_duration:.2f} seconds\n")

    return partitions


def parse_benchmark_results_xlsx(file_path):
    """
    Parse the benchmark results from an Excel (.xlsx) file using pandas.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        dict: A dictionary where keys are test case names and values are execution times.
    """
    try:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(file_path)

        # Ensure required columns exist
        if "Testcase Name" not in df.columns or "Execution Time (s)" not in df.columns:
            raise ValueError(
                "The Excel file must contain 'Testcase Name' and 'Execution Time (s)' columns."
            )

        # Convert the DataFrame to a dictionary
        results = df.set_index("Testcase Name")["Execution Time (s)"].to_dict()

        return results

    except Exception as e:
        print(f"Error parsing benchmark results: {e}")
        return {}


def generate_formatted_test_matrix_from_partitions(
    partitions, base_name="bmk", runs_on="wormhole_b0"
):
    matrix = []
    splits = []  # 65K limit on string size
    for i, partition in enumerate(partitions):
        job_name = f"{base_name}_{i}"

        # test duration is reported in seconds
        expected_duration_s = 0
        for testcase in partition:
            expected_duration_s += testcase["test-duration"]

        # Timeout logic for progressive timeout setting.
        # 1.  If there is no known duration for the test, set the timeout to a reasonable default of 2 hours
        # 2.  If a test is expected to take less than 30 minutes, set the timeout to 1 hour.
        # 3.  If the expected duration exceeds 30 minutes, set it to the MAX_TIMEOUT of 500m

        timeout_saturation_threshold_s = 30 * 60
        actual_timeout = 0

        if expected_duration_s < 0:
            actual_timeout = DEFAULT_JOB_TIMEOUT_MINUTES
        elif expected_duration_s < timeout_saturation_threshold_s:
            actual_timeout = 60
        else:
            actual_timeout = MAXIMUM_JOB_TIMEOUT_MINUTES

        # Append the test name to the job name for quarantined tests
        if len(partition) == 1:
            # sanitize partition names.
            job_name += "_qtn_" + re.sub(
                r"[^\w\-]", "_", partition[0]["full-test-name"].split("::")[-1]
            )
        matrix.append(
            {
                "tests": partition,
            }
        )
        splits.append(
            {
                "runs-on": runs_on,
                "name": job_name,
                "group-id": i,  # zero indexed
                "t-o": actual_timeout,  # timeout in minutes, for timeout-minutes field
            }
        )

    # we cannot pass the matrix JSON string directly as a job output due to output string size limits
    return json.dumps(matrix), json.dumps(splits)


def get_existing_execute_tests():
    # We don't care about tests that already run in execute.
    # Benchmarks tell us:
    #   if a model regressed (eg. expect to run in execute, but now run in compile)
    #   if a model is promotable (eg. expect to run in compile, runs in execute)
    # Regression testing here is redundant to full model exec tests in nightly
    # Models cannot be promoted beyond execute so we don't care about promotability either

    exec_tests = []
    exec_tests = parse_tests_from_matrix(
        ".github/workflows/run-full-model-execution-tests.yml"
    )
    exec_tests.extend(
        parse_tests_from_matrix(
            ".github/workflows/run-full-model-execution-tests-nightly.yml"
        )
    )
    return exec_tests


def generate_dynamic_benchmark_test_matrix():

    output_file = "benchmark_test_matrix.json"  # hardcoded into CI
    output_file_splits = "benchmark_test_matrix_splits.json"  # hardcoded into CI
    report_dir = "benchmark_report"
    report_branch = "benchmark_report_fs"

    download_artifact_command = [
        "python",
        "results/download_artifacts.py",
        "--workflow",
        "upload-benchmark-file.yml",
        "--filter",
        "xlsx",
        "--branch",
        report_branch,
        "-o",
        report_dir,
    ]
    print("Running download artifact command: ", " ".join(download_artifact_command))
    # download previous report from reporting path
    subprocess.run(
        download_artifact_command,
        check=True,
        shell=False,
    )

    # parse the xlsx file to get known test execution times
    reports = os.listdir(report_dir)
    assert (
        len(reports) == 1
    ), f"Expected exactly one xlsx file in {report_dir}, found: {reports}"
    previous_run_results = parse_benchmark_results_xlsx(report_dir + "/" + reports[0])

    # All pytests defined in tests/models
    in_tree_tests = enumerate_all_tests()

    actual_test_durations_list = {}
    quarantined_tests = []

    for test in in_tree_tests:

        # Workaround to prevent quarantine of renamed tests with red/generality suffixing
        modified_test_name = test.replace("_red", "").replace("_generality", "")

        if test in previous_run_results.keys():
            actual_test_durations_list[test] = previous_run_results[test]
        if modified_test_name in previous_run_results.keys():
            # Handle cases where the test name in the report may be slightly different
            actual_test_durations_list[test] = previous_run_results[modified_test_name]
        else:
            # quarantined testss
            actual_test_durations_list[test] = -1
            quarantined_tests.append(test)

    print(f"Quarantined test list (ct: {len(quarantined_tests)})")
    print(f"Actual test list (ct: {len(actual_test_durations_list)})")

    # Remove all pytests already assessed in full model exec tests
    already_executing_tests = get_existing_execute_tests()

    initial_count = len(actual_test_durations_list)
    # Prune the actual_test_durations_list
    actual_test_durations_list = {
        test: duration
        for test, duration in actual_test_durations_list.items()
        if test not in already_executing_tests
    }
    print(
        f"{initial_count - len(actual_test_durations_list)} tests removed from the test list because they already are tested in EXECUTE"
    )

    # Load balance the tests into a dynamic test matrix
    test_splits = load_balance_tests_greedy(
        actual_test_durations_list, do_linearize=True
    )
    fmt_matrix, splits_matrix = generate_formatted_test_matrix_from_partitions(
        test_splits
    )
    print("Formatted test matrix:")
    print(fmt_matrix)
    print("Test splits matrix")
    print(splits_matrix)

    # Write the matrix to a file
    with open(output_file, "w") as f:
        f.write(fmt_matrix)
    print(f"Test matrix written to {output_file}")

    # Write the splits to a file.

    with open(output_file_splits, "w") as f:
        f.write(splits_matrix)
    print(f"Test matrix written to {output_file_splits}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utilities for new pytest promotion to benchmarks"
    )
    parser.add_argument(
        "--discover-tests",
        action="store_true",
        help="Discover tests in tree and compare with benchmark tests.",
    )

    parser.add_argument(
        "--gen-matrix",
        action="store_true",
        help="Dynamically generate the test matrix with quarantined tests",
    )

    args = parser.parse_args()
    if args.discover_tests:
        """
        Discover tests in the tree and compare with benchmark tests.
        """
        discover_tests()

    if args.gen_matrix:
        # new_tests = discover_tests()
        generate_dynamic_benchmark_test_matrix()
