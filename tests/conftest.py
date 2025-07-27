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


# Memory Profiling
def get_peak_rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                kb = int(line.split()[1])
                return kb / 1024 / 1024
    return 0


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_call(item):
    yield
    peak_rss = get_peak_rss_gb()
    print(f"\n[PEAK MEMORY] {item.nodeid}: {peak_rss:.2f} GB", flush=True)
