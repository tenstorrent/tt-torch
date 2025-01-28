# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.fixture(autouse=True)
def run_around_tests():
    torch.manual_seed(0)
    yield
    torch._dynamo.reset()


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
