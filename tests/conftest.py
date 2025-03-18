# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import subprocess
import sys
from datetime import datetime, timezone
from tt_torch.tools.utils import OpByOpBackend
from tt_torch.tools.crashsafe_utils import crashsafe_suffix

import os
import json
import shutil

global junitxml_path
junitxml_path = None


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
        "--op_by_op_stablehlo",
        action="store_true",
        default=False,
        help="Run test in stablehlo op-by-op mode",
    )
    parser.addoption(
        "--op_by_op_torch",
        action="store_true",
        default=False,
        help="Run test in torch op-by-op mode",
    )
    parser.addoption(
        "--crashsafe",
        action="store_true",
        default=False,
        help="Create transacted output logs",
    )


def pytest_collection_modifyitems(config, items):
    # Filter tests based on which op_by_op flag is set
    selected_items = []
    using_torch = config.getoption("--op_by_op_torch")
    using_stablehlo = config.getoption("--op_by_op_stablehlo")

    for item in items:
        for param in item.iter_markers(name="parametrize"):
            if "op_by_op" in param.args[0]:
                op_by_op_value = item.callspec.params["op_by_op"]
                # Only select tests that match the specific backend flag
                if (using_torch and op_by_op_value == OpByOpBackend.TORCH) or (
                    using_stablehlo and op_by_op_value == OpByOpBackend.STABLEHLO
                ):
                    selected_items.append(item)
                    break

    if using_torch or using_stablehlo:
        # Replace the items with only the selected backend tests
        items[:] = selected_items


@pytest.fixture(scope="function", autouse=True)
def record_test_timestamp(record_property):
    start_timestamp = datetime.now(timezone.utc).isoformat()
    record_property("start_timestamp", start_timestamp)
    yield
    end_timestamp = datetime.now(timezone.utc).isoformat()
    record_property("end_timestamp", end_timestamp)


@pytest.fixture
def record_property(request):
    """Override the built-in record_property fixture with transactional writes."""
    global junitxml_path

    def _original_record_property(name, value):
        # Copied from https://docs.pytest.org/en/7.1.x/_modules/_pytest/junitxml.html
        request.node.user_properties.append((name, value))

    # config options not in scope at this point, so using this global as indicator
    if not junitxml_path:
        return _original_record_property

    property_file = f"{junitxml_path}{crashsafe_suffix}"
    print(f"Writing to {property_file}")
    with open(property_file, "w+") as f:
        json.dump({}, f)

    def _crashsafe_record_property(name, value):
        # Add to the standard pytest user_properties
        request.node.user_properties.append((name, value))

        # Also write to disk immediately for durability
        properties = {}
        if os.path.exists(property_file):
            with open(property_file, "r") as f:
                try:
                    properties = json.load(f)
                except json.JSONDecodeError:
                    properties = {}

        properties[name] = value

        # Write atomically to avoid corruption
        temp_file = f"{property_file}.tmp"
        with open(temp_file, "w") as f:
            json.dump(properties, f)
        os.replace(temp_file, property_file)

    return _crashsafe_record_property


def pytest_configure(config):
    global junitxml_path

    # Check if the --crashsafe option is enabled
    if config.getoption("--crashsafe"):
        print(f"Running in crashsafe mode - logging data to crashsafe log")
        junitxml_path = config.getoption("--junit-xml")
