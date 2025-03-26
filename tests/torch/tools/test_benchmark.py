# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
from tt_torch.tools.crashsafe_utils import crashsafe_suffix, get_achieved_compile_depths
from tt_torch.tools.generate_benchmark_report import (
    parse_benchmark_xml,
    achieved_depth_mapping,
)
import subprocess
import shutil
import signal


def insert_digit_into_filename(filename, digit):
    """
    Inserts a digit into the filename before the last file extension.

    Args:
        filename (str): The original filename.
        digit (int): The digit to insert.

    Returns:
        str: The modified filename.
    """
    base, ext = os.path.splitext(filename)
    return f"{base}_{digit}{ext}"


def test_crashsafe_utils():
    """
    This tests the functionality of the crashsafe report generation and merging utils:
    1. Run a normal test with crashsafe enabled
    2. Run a different test with crashsafe enabled and sigkill it when it passes the constructor to simulate a model crashing during EXECUTE
    3. Check if the crashsafe reports are generated
    4. Duplicate the normal crashsafe report n_dupes (5) times to simulate multiple models
    5. Merge the crashsafe_reports using the fuser script: postprocess_crashsafe_reports
    6. Check if the merged report is generated
    7. Parse the merged report with the benchmark parser
    8. Check if the number of models parsed in the merge report is correct
    9. Check if the parsed benchmark data has valid structure
    """

    test_name = "tests/models/autoencoder_linear/test_autoencoder_linear.py::test_autoencoder_linear[full-eval]"
    sigkilled_test_name = (
        "tests/models/mnist/test_mnist.py::test_mnist_train[full-eval]"
    )

    report_dir = "results/__tmp__crashsafe_test/"
    try:
        if not os.path.exists(report_dir):
            os.mkdir(report_dir)

        merged_report_dir = report_dir + "merged/"
        if not os.path.exists(merged_report_dir):
            os.mkdir(merged_report_dir)

        report_name = report_dir + "autoencoder_linear.xml"
        sigkilled_report_name = report_dir + "mnist.xml"

        crashsafe_name = report_name + crashsafe_suffix
        merged_report_name = merged_report_dir + "merged_report.xml"

        run_command = ["-svv", "--crashsafe", f"--junit-xml={report_name}", test_name]

        try_sigkill_safe_logging(
            [
                "pytest",
                "-svv",
                "--crashsafe",
                f"--junit-xml={sigkilled_report_name}",
                sigkilled_test_name,
            ]
        )

        pytest.main(["-svv", "--crashsafe", f"--junit-xml={report_name}", test_name])

        # check if crashsafe report is generated
        assert os.path.exists(
            crashsafe_name
        ), f"Crashsafe report not generated, expected at {crashsafe_name}"

        # check crashsafe report format
        compile_depths = get_achieved_compile_depths(crashsafe_name)
        print(compile_depths)

        assert compile_depths, "No compile depths found in crashsafe report"

        # duplicate the individual report
        n_dupes = 5
        for i in range(n_dupes):
            subprocess.run(
                f"cp {crashsafe_name} {insert_digit_into_filename(crashsafe_name, i)}",
                shell=True,
                check=True,
            )

        # test crashsafe report fusion
        merge_command = f'python tt_torch/tools/postprocess_crashsafe_reports.py "{report_dir}/*crashsafe*.xml" "{merged_report_name}"'
        print("Running merge command:")
        print(merge_command)
        subprocess.run(merge_command, shell=True, check=True)
        assert os.path.exists(
            merged_report_name
        ), f"Fusedf report not generated, expected at {merged_report_name}"

        # check for 1+n_dupes models in the merged report
        bm_results = parse_benchmark_xml(merged_report_dir)
        print("Benchmark results:")
        # +2 for original test and sigkilled test
        assert (
            len(bm_results) == 2 + n_dupes
        ), f"Expected {2 + n_dupes} models in the merged report, found {len(bm_results)}"

        # check structure of results
        for result in bm_results:
            model_name = result[0]
            execution_time = result[1]
            compile_depth = result[2]
            print(model_name, execution_time, compile_depth)
            assert (
                "::" in model_name
            ), f"Model name {model_name} does not contain '::' to indicate it is a pytest"
            assert (
                execution_time >= 0
            ), f"Execution time {execution_time} is not a valid duration"
            assert (
                compile_depth in achieved_depth_mapping.keys()
            ), f"Compile depth {compile_depth} is not a valid compile depth"
    except Exception as e:
        raise e
    finally:
        # cleanup files
        print(f"Cleaning up test files in {report_dir}")
        shutil.rmtree(report_dir)


def try_sigkill_safe_logging(test_command):
    search_string = "SiliconDriver"  # We know that we passed the constructor when we see this loguru printout: 2025-03-26 20:04:47.944 | INFO     | SiliconDriver   - Opened PCI device 6; KMD version: 1.31.0, IOMMU: disabled

    process = subprocess.Popen(
        test_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        # Monitor the stdout line by line
        for line in process.stdout:
            print(line, end="")  # Print the output for visibility
            if search_string in line:
                print(
                    f"Detected '{search_string}' in stdout. Sending SIGKILL to the process."
                )
                process.kill()  # Send SIGKILL to terminate the process
                break

        # Wait for the process to terminate
        process.wait()
    except Exception as e:
        print(f"An error occurred: {e}")
        process.kill()  # Ensure the process is terminated in case of an error
    finally:
        # Cleanup: Close stdout and stderr
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()
