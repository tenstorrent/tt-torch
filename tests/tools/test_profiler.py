# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import tt_torch
import subprocess
import os
from tt_torch.tools.profile import profile
from tt_torch.tools.perf import Profiler

test_command = (
    "pytest -svv tests/models/mnist/test_mnist.py::test_mnist_train[full-eval]"
)
expected_output_location = f"{Profiler.get_ttmetal_home_path()}/generated/profiler/reports/{Profiler.DEFAULT_OUTPUT_FILENAME}"


def test_profiler_cli():
    profiler_command = f'python tt_torch/tools/profile.py "{test_command}"'
    profiler_subprocess = subprocess.run(profiler_command, shell=True)

    # Check return code
    assert (
        profiler_subprocess.returncode == 0
    ), "Profiler exited with non-zero return code {profier_subprocess.returncode}"

    # Check that the profiler generated the expected files
    assert os.path.exists(
        expected_output_location
    ), f"Result file @ {expected_output_location} not found"


def test_profiler_module():
    profile(test_command)

    assert os.path.exists(
        expected_output_location
    ), f"Result file @ {expected_output_location} not found"
