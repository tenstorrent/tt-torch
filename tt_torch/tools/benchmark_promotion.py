# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re
import subprocess
from tt_torch.tools.generate_benchmark_report import parse_tests_from_matrix
from collections import Counter
import csv


def enumerate_all_tests():
    test_dir = "tests/models"
    try:
        # Run pytest with --collect-only and capture the output
        # sudo apt install -y libgl1 libglx-mesa0 # (may be needed locally)
        result = subprocess.run(
            ["pytest", test_dir, "--collect-only", "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Check if pytest ran successfully
        if result.returncode != 0:
            raise RuntimeError(f"pytest failed: {result.stderr}")

        # Extract test names using a regex
        test_cases = re.findall(r"(\S+::\S+)", result.stdout)

        return [tc for tc in test_cases if "full" in tc and "eval" in tc]

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

    print("Identifying changed pytest test set.")
    removed_tests = set(benchmark_tests) - set(in_tree_tests)
    added_tests = set(in_tree_tests) - set(benchmark_tests)

    if removed_tests != set():
        print(f"Warning {len(removed_tests)} pytests have been removed from tree.")

    # Added tests should be quarantined into their own runners and run in isolation
    # so unexpected/unrecoverable breakage doesn't cause other benchmarks to fail
    if (added_tests) != set():
        print(
            f"{len(added_tests)} pytests have been added to tree and are not in the current benchmark."
        )
        for i, t in enumerate(added_tests):
            print(f"{i+1}\t{t}")


def load_balance_tests_greedy(csv_file_path, n_partitions):
    """
    Load balances test names from a CSV file into N partitions based on test execution time.

    Args:
        csv_file_path (str): Path to the CSV file containing test data.
        n_partitions (int): Number of partitions to split the test names into.

    Returns:
        list of lists: A list containing N partitions, each with a subset of test names.
    """
    # Read the CSV file and extract test names and their execution times
    tests = []
    with open(csv_file_path, mode="r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            test_name = row["full_test_name"]
            test_time = float(row["test_execution_time"])
            tests.append((test_name, test_time))

    # Sort tests by execution time in descending order
    tests.sort(key=lambda x: x[1], reverse=True)
    print(f"Found {len(tests)} tests in {csv_file_path}")

    # Initialize partitions and their total times
    partitions = [[] for _ in range(n_partitions)]
    partition_times = [0] * n_partitions

    # Distribute tests greedily to minimize the maximum partition time
    for test_name, test_time in tests:
        # Find the partition with the smallest total time
        min_index = partition_times.index(min(partition_times))
        partitions[min_index].append((test_name, test_time))
        partition_times[min_index] += test_time

    return partitions


if __name__ == "__main__":
    # Example usage
    csv_file_path = ".github/benchmark_durations.csv"
    n_partitions = 4

    partitions = load_balance_tests_greedy(csv_file_path, n_partitions)
    for i, partition in enumerate(partitions):
        print(
            f"Partition {i + 1}: {len(partition)} tests, Total Time: {sum(float(row[1])/3600 for row in partition):.2f}h"
        )
        print(
            partition[:5]
        )  # Print the first 5 test names in each partition for brevity

    formatted_partitions = [[pair[0] for pair in p] for p in partitions]
    print(formatted_partitions)
