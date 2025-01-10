# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import multiprocessing as mp

mp.set_start_method("spawn")


@pytest.fixture(autouse=True)
def run_around_tests():
    torch.manual_seed(0)
    yield
    torch._dynamo.reset()


def pytest_addoption(parser):
    parser.addoption(
        "--nightly",
        action="store_true",
        default=False,
    )


def pytest_generate_tests(metafunc):
    if "nightly" in metafunc.fixturenames:
        # Retrieve value of the custom option
        value = metafunc.config.getoption("nightly")
        metafunc.parametrize("nightly", [value])
