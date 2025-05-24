# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import tt_torch
import subprocess
import os
from tt_torch.tools.profile import profile
from tt_torch.tools.profile_util import Profiler
import csv

test_command_mnist = "pytest -svv tests/models/mnist/test_mnist.py::test_mnist_train[full-eval-single_device]"
test_command_add = "pytest -svv tests/torch/test_basic.py::test_add"
expected_report_path = f"results/perf/{Profiler.DEFAULT_OUTPUT_FILENAME}"


def test_profiler_cli():
    profiler_command = f'python tt_torch/tools/profile.py "{test_command_add}"'
    profiler_subprocess = subprocess.run(profiler_command, shell=True)

    # Check return code
    assert (
        profiler_subprocess.returncode == 0
    ), f"Profiler exited with non-zero return code {profiler_subprocess.returncode}"

    # Check that the profiler generated the expected files
    assert os.path.exists(
        expected_report_path
    ), f"Result file @ {expected_report_path} not found"

    with open(expected_report_path, "r") as f:
        reader = csv.DictReader(f)
        assert (
            "LOC" in reader.fieldnames
        ), f"Profiler output does not contain expected output"

        print("Perf Report Contents")
        for row in reader:
            print(row)


def test_profiler_module():
    profile(test_command_add)

    assert os.path.exists(
        expected_report_path
    ), f"Result file @ {expected_report_path} not found"

    with open(expected_report_path, "r") as f:
        reader = csv.DictReader(f)
        assert (
            "LOC" in reader.fieldnames
        ), f"Profiler output does not contain expected output"

        print("Perf Report Contents")
        for row in reader:
            print(row)
