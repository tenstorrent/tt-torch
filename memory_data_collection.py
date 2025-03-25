#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import time
import signal
import subprocess
import psutil
import argparse
import csv
import datetime
from pathlib import Path


def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB


def run_test(test_name, iterations, output_dir, username):
    """Run pytest for specific test multiple times and track memory usage"""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare CSV file for results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"{test_name}_{timestamp}.csv")

    # Initialize CSV with headers
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "iteration",
                "start_memory_mb",
                "peak_memory_mb",
                "end_memory_mb",
                "execution_time_s",
                "status",
            ]
        )

    # Signal handler to ensure we save data on interruption
    def signal_handler(sig, frame):
        print(f"\nProcess interrupted. Results saved to {csv_path}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run tests for the specified number of iterations
    for i in range(1, iterations + 1):
        print(f"\n{'='*50}")
        print(f"Running {test_name} - Iteration {i}/{iterations}")
        print(f"{'='*50}")

        # Measure starting memory
        start_memory = get_memory_usage()
        start_time = time.time()
        peak_memory = start_memory

        # Run the pytest command
        command = f"pytest -svv tests/models/{test_name} --nightly"

        try:
            # Execute the pytest
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Monitor memory usage during test execution
            while process.poll() is None:
                current_memory = get_memory_usage()
                peak_memory = max(peak_memory, current_memory)
                # Print progress to console
                if process.stdout:
                    for line in process.stdout:
                        print(line, end="")
                time.sleep(0.1)

            # Get the exit code
            exit_code = process.returncode
            status = "PASS" if exit_code == 0 else f"FAIL (code: {exit_code})"

        except Exception as e:
            status = f"ERROR: {str(e)}"
            exit_code = -1

        # Calculate execution time and final memory
        execution_time = time.time() - start_time
        end_memory = get_memory_usage()

        # Save results to CSV
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    i,
                    round(start_memory, 2),
                    round(peak_memory, 2),
                    round(end_memory, 2),
                    round(execution_time, 2),
                    status,
                ]
            )

        # Clean cache
        print(f"\nCleaning cache: rm -rf /localdev/{username}/cache/*")
        try:
            subprocess.run(
                f"rm -rf /localdev/{username}/cache/*", shell=True, check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to clean cache: {e}")

        print(f"Iteration {i} completed in {execution_time:.2f}s")
        print(
            f"Memory usage: Start: {start_memory:.2f}MB, Peak: {peak_memory:.2f}MB, End: {end_memory:.2f}MB"
        )
        print(f"Status: {status}")

        # Add a small delay between iterations
        if i < iterations:
            time.sleep(1)

    print(f"\nAll iterations completed. Results saved to {csv_path}")
    return csv_path


def run_all_tests(test_names, iterations, output_dir, username):
    """Run all specified tests"""
    results = {}

    for test_name in test_names:
        print(f"\n{'#'*60}")
        print(f"Starting test suite: {test_name}")
        print(f"{'#'*60}")

        csv_path = run_test(test_name, iterations, output_dir, username)
        results[test_name] = csv_path

    print("\nSummary of all test runs:")
    for test_name, csv_path in results.items():
        print(f"{test_name}: Results saved to {csv_path}")


USERNAME = "ddilbaz"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run pytests multiple times and track memory usage"
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["clip", "flan_t5", "vilt", "codegen", "RMBG", "mgp-str-base"],
        help="List of test names to run",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations to run each test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./memory_results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    run_all_tests(args.tests, args.iterations, args.output_dir, USERNAME)
