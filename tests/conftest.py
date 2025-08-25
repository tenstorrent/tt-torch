# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import subprocess
import sys
from datetime import datetime, timezone
from tt_torch.tools.utils import OpByOpBackend
from tt_torch.tools.crashsafe_utils import crashsafe_suffix
import xml.etree.ElementTree as ET
import socket
import os
import tt_mlir
import tt_torch.dynamo.sharding_utils as sharding_utils

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


@pytest.fixture(scope="function")
def use_xla_spmd_environment():
    """
    Setup XLA environment for tensor parallelism.
    SPMD and nonSPMD tests are not to be mixed in the same pytest process/test group.
    """

    env_vars_to_restore = {
        "TT_TORCH_FORCE_EXPERIMENTAL_BACKEND": os.environ.get(
            "TT_TORCH_FORCE_EXPERIMENTAL_BACKEND", None
        )
    }

    # Needed to set TT_TORCH_FORCE_EXPERIMENTAL_BACKEND to reuse verify_torch_module
    os.environ["TT_TORCH_FORCE_EXPERIMENTAL_BACKEND"] = "1"
    sharding_utils.setup_xla_spmd_environment()

    yield

    # Restore original environment variable values
    for var_name, original_value in env_vars_to_restore.items():
        if original_value is None:
            os.environ.pop(var_name, None)
        else:
            os.environ[var_name] = original_value


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
    parser.addoption(
        "--log-memory-usage",
        action="store_true",
        default=False,
        help="log per-test memory usage into pytest-memory-usage.csv",
    )


# Print more details for skipped and xfailed tests
def pytest_runtest_logreport(report):
    if report.outcome in ("xfailed", "skipped") or hasattr(report, "wasxfail"):
        if report.longrepr:
            print(f"\n{report.longreprtext}")


def pytest_collection_modifyitems(config, items):
    # Filter tests based on which op_by_op flag is set
    selected_items = []
    using_torch = config.getoption("--op_by_op_torch")
    using_stablehlo = config.getoption("--op_by_op_stablehlo")

    # Check if the --crashsafe option is enabled
    if config.getoption("--crashsafe"):
        # Count the number of collected tests
        num_tests = len(items)
        if num_tests > 1:
            pytest.exit(
                f"Error: --crashsafe can only be used with a single test. {num_tests} tests were specified.",
                returncode=1,
            )

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
    """Override the built-in record_property fixture with transactional writes to XML."""
    global junitxml_path

    def _original_record_property(name, value):
        # Copied from https://docs.pytest.org/en/7.1.x/_modules/_pytest/junitxml.html
        request.node.user_properties.append((name, value))

    # If junitxml_path is not set, fallback to the original behavior
    if not junitxml_path:
        return _original_record_property

    property_file = f"{junitxml_path}{crashsafe_suffix}"
    print(f"Writing to {property_file}")

    def _crashsafe_record_property(name, value):
        # Add to the standard pytest user_properties
        request.node.user_properties.append((name, value))

        # Load the existing XML file
        tree = ET.parse(property_file)
        root = tree.getroot()

        # Find or create a <testsuite> element for the current test
        test_name = request.node.nodeid
        testsuite = root.find(f"./testsuite[@name='pytest']")
        if testsuite is None:
            testsuite = ET.SubElement(
                root,
                "testsuite",
                name="pytest",
                errors="0",
                failures="0",
                skipped="0",
                tests="1",
                time="0",  # Placeholder, can be updated later
                timestamp=datetime.now().isoformat(),
                hostname=socket.gethostname(),
            )

        # Find or create a <testcase> element for the current test
        testcase = testsuite.find(f"./testcase[@name='{test_name}']")
        if testcase is None:
            testcase = ET.SubElement(
                testsuite,
                "testcase",
                classname=request.node.module.__name__,
                name=test_name,
                time="0",  # Placeholder, can be updated later
            )

        # Add the property to the <testcase>
        properties = testcase.find("properties")
        if properties is None:
            properties = ET.SubElement(testcase, "properties")

        # Ensure the value is properly formatted (e.g., JSON for complex objects)
        if isinstance(value, (dict, list)):
            value = str(value)

        ET.SubElement(properties, "property", name=name, value=value)

        # Write the updated XML back to the file
        temp_file = f"{property_file}.tmp"
        tree.write(temp_file, encoding="utf-8", xml_declaration=True)

        os.replace(temp_file, property_file)

    return _crashsafe_record_property


def pytest_configure(config):
    global junitxml_path

    # Check if the --crashsafe option is enabled
    if config.getoption("--crashsafe"):
        junitxml_path = config.getoption("--junit-xml")
        property_file = f"{junitxml_path}{crashsafe_suffix}"
        print(f"Writing crashsafe log to {property_file}")

        # Ensure the XML file exists and has a root element
        root = ET.Element("testsuites")
        tree = ET.ElementTree(root)
        tree.write(property_file)


# This is lifted from tt-forge-fe mostly.

import psutil
import time
import gc


@pytest.fixture(autouse=True)
def memory_usage_tracker_and_cleanup(request):
    """
    A pytest fixture that tracks memory usage during the execution of a test and cleans up after the test.

    This fixture automatically tracks the memory usage of the process running the tests.
    It starts tracking before the test runs, continues tracking in a background thread during the test,
    and stops tracking after the test completes. It logs the memory usage statistics including the
    minimum, maximum, average, and total memory usage by the test.

    The memory usage is measured in megabytes (MB).

    Note:
        - This fixture is automatically used for all tests due to the `autouse=True` parameter.
        - The interval for memory readings can be adjusted by changing the sleep duration in the `track_memory` function.
        - Min, max, and avg memory usage are calculated based on the recorded memory readings from system memory.
        - After the test completes, the fixture performs cleanup: it runs Python garbage collection and calls
          `tt_mlir.malloc_trim()` to release memory back to the OS when possible.
        - Verbose console printing is controlled via the `TT_TORCH_VERBOSE_MEMORY_TRACKER` environment variable.
          Set it to `1` to enable prints; set to `0` or unset to suppress prints.
    """
    process = psutil.Process()

    # Get the current test name
    test_name = request.node.name

    # Initialize memory tracking variables
    start_mem = process.memory_info().rss / (1024 * 1024)  # MB
    verbose = os.getenv("TT_TORCH_VERBOSE_MEMORY_TRACKER", "0") == "1"

    min_mem = start_mem
    max_mem = start_mem
    total_mem = start_mem
    count = 1

    # Start a background thread or loop to collect memory usage over time
    tracking = True

    def track_memory():
        nonlocal min_mem, max_mem, total_mem, count
        while tracking:
            current_mem = process.memory_info().rss / (1024 * 1024)
            min_mem = min(min_mem, current_mem)
            max_mem = max(max_mem, current_mem)
            total_mem += current_mem
            count += 1
            time.sleep(0.1)  # Adjust the interval as needed

    # Start tracking in a background thread
    import threading

    tracker_thread = threading.Thread(target=track_memory)
    tracker_thread.start()

    # Run the test
    yield

    # Stop tracking and wait for the thread to finish
    tracking = False
    tracker_thread.join()

    # Calculate end memory and memory usage stats
    end_mem = process.memory_info().rss / (1024 * 1024)  # MB
    min_mem = min(min_mem, end_mem)
    max_mem = max(max_mem, end_mem)
    total_mem += end_mem
    count += 1
    avg_mem = total_mem / count

    by_test = max_mem - start_mem

    before_gc = process.memory_info().rss / (1024 * 1024)
    gc.collect()  # Force garbage collection
    after_gc = process.memory_info().rss / (1024 * 1024)
    tt_mlir.malloc_trim()
    after_trim = process.memory_info().rss / (1024 * 1024)

    # Log memory usage statistics
    if verbose:
        print(f"\nTest memory usage for {test_name}:")
        print(f"    By test: {by_test:.2f} MB")
        print(f"    Minimum: {min_mem:.2f} MB")
        print(f"    Maximum: {max_mem:.2f} MB")
        print(f"    Average: {avg_mem:.2f} MB")
        print(f"Memory usage before test: {start_mem:.2f} MB")
        print(f"Memory usage before garbage collection: {before_gc:.2f} MB")
        print(f"Memory usage after garbage collection: {after_gc:.2f} MB")
        print(f"Memory usage after malloc_trim: {after_trim:.2f} MB")

    should_log = request.config.getoption("--log-memory-usage")
    if not should_log:
        return

    # Store memory usage stats into a CSV file
    file_name = "pytest-memory-usage.csv"
    with open(file_name, "a") as f:
        if f.tell() == 0:
            # Write header if file is empty
            f.write(
                "test_name,start_mem,end_mem,min_memory,max_memory,by_test (approx), after_gc, after_trim\n"
            )
        # NOTE: escape test_name in double quotes because some tests have commas in their parameter list...
        f.write(
            f'"{test_name}",{start_mem:.2f},{end_mem:.2f},{min_mem:.2f},{max_mem:.2f},{by_test:.2f},{after_gc:.2f},{after_trim:.2f}\n'
        )
