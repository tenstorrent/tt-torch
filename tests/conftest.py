# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import subprocess
import sys
from datetime import datetime, timezone


@pytest.fixture(autouse=True)
def run_around_tests():
    torch.manual_seed(0)
    yield
    torch._dynamo.reset()


@pytest.fixture(scope="module")
def manage_dependencies(request):
    dependencies = getattr(request.module, "dependencies", [])
    # Install dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + dependencies)
    yield
    # Uninstall dependencies
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y"] + dependencies
    )


def pytest_addoption(parser):
    parser.addoption(
        "--op_by_op",
        action="store_true",
        default=False,
        help="Run test in op-by-op mode",
    )


def pytest_collection_modifyitems(config, items):
    # If --op_by_op flag is set, filter out tests with op_by_op=False
    selected_items = []
    for item in items:
        # Check if the test has a parameter called 'op_by_op'
        # and whether it is set to True

        if config.getoption("--op_by_op"):
            for param in item.iter_markers(name="parametrize"):
                # Check if the parameter is 'op_by_op' and its value is True
                if "op_by_op" in param.args[0] and item.callspec.params["op_by_op"]:
                    selected_items.append(item)
                    break
        else:
            # If the test does not have a 'op_by_op' parameter,
            # add all tests without a op_by_op parameter, as well
            # as all testst with a op_by_op parameter where op_by_op=False
            has_op_by_op_param = False
            for param in item.iter_markers(name="parametrize"):
                if "op_by_op" in param.args[0]:
                    has_op_by_op_param = True
                    # Only add the test if op_by_op=False
                    if not item.callspec.params["op_by_op"]:
                        selected_items.append(item)
                        break
            # If theres no nigtly parameter, add the test
            if not has_op_by_op_param:
                selected_items.append(item)
        # Replace the items with only the op_by_op tests
    items[:] = selected_items


@pytest.fixture(scope="function", autouse=True)
def record_test_timestamp(record_property):
    start_timestamp = datetime.now(timezone.utc).isoformat()
    record_property("start_timestamp", start_timestamp)
    yield
    end_timestamp = datetime.now(timezone.utc).isoformat()
    record_property("end_timestamp", end_timestamp)
