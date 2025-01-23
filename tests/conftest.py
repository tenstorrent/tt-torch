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
        "--nightly", action="store_true", default=False, help="Run nightly tests"
    )


def pytest_collection_modifyitems(config, items):
    # If --nightly flag is set, filter out tests with nightly=False
    selected_items = []
    for item in items:
        # Check if the test has a parameter called 'nightly'
        # and whether it is set to True

        if config.getoption("--nightly"):
            for param in item.iter_markers(name="parametrize"):
                # Check if the parameter is 'nightly' and its value is True
                if "nightly" in param.args[0] and item.callspec.params["nightly"]:
                    selected_items.append(item)
                    break
        else:
            # If the test does not have a 'nightly' parameter,
            # add all tests without a nightly parameter, as well
            # as all testst with a nightly parameter where nightly=False
            has_nightly_param = False
            for param in item.iter_markers(name="parametrize"):
                if "nightly" in param.args[0]:
                    has_nightly_param = True
                    # Only add the test if nightly=False
                    if not item.callspec.params["nightly"]:
                        selected_items.append(item)
                        break
            # If theres no nigtly parameter, add the test
            if not has_nightly_param:
                selected_items.append(item)
        # Replace the items with only the nightly tests
    items[:] = selected_items
