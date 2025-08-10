# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.runner.test_config import test_config
from tests.runner.test_utils import ModelTestConfig, ModelStatus
import difflib
import signal
import json
import os

# Global set to track collected test node IDs
_collected_nodeids = set()

# Set of full nodeids that have known durations (from .test_durations)
_tests_with_known_durations = set()


def pytest_addoption(parser):
    parser.addoption(
        "--arch",
        action="store",
        default=None,
        help="Target architecture (e.g., blackhole, wormhole) for which to match via arch_overrides in test_config.py",
    )
    parser.addoption(
        "--validate-test-config",
        action="store_true",
        default=False,
        help="Fail if test_config.py and collected test IDs are out of sync",
    )
    parser.addoption(
        "--timeout-seconds",
        action="store",
        type=int,
        default=1800,
        help="Default global per-test timeout in seconds (default: 1800)",
    )


def pytest_configure(config):
    # Register custom marker to avoid PytestUnknownMarkWarning
    config.addinivalue_line("markers", "timeout(seconds): Per-test timeout in seconds")

    # Load known durations from .test_durations if present
    durations_path = os.path.join(os.getcwd(), ".test_durations")
    try:
        with open(durations_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                _tests_with_known_durations.update(data.keys())
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        # If file exists but is not JSON, ignore
        pass


@pytest.fixture(autouse=True)
def log_test_name(request):
    print(f"\nRunning {request.node.nodeid}", flush=True)


@pytest.fixture
def test_metadata(request) -> ModelTestConfig:
    meta = getattr(request.node, "_test_meta", None)
    assert meta is not None, f"No ModelTestConfig attached for {request.node.nodeid}"
    return meta


# Enforce a global timeout unless the test is present in .test_durations
@pytest.fixture(autouse=True)
def enforce_global_timeout(request):
    # Skip timeout if this test has a known duration listed
    if request.node.nodeid in _tests_with_known_durations:
        yield
        return

    # Allow override via explicit marker; otherwise use the global default
    marker = request.node.get_closest_marker("timeout")
    if marker is not None:
        if marker.kwargs.get("seconds") is not None:
            seconds = int(marker.kwargs["seconds"])
        elif marker.args:
            seconds = int(marker.args[0])
        else:
            seconds = request.config.getoption("--timeout-seconds")
    else:
        seconds = request.config.getoption("--timeout-seconds")

    def _handle_timeout(signum, frame):
        pytest.fail(
            f"Test exceeded timeout of {seconds} seconds: {request.node.nodeid}",
            pytrace=False,
        )

    previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def pytest_collection_modifyitems(config, items):
    arch = config.getoption("--arch")
    validate_config = config.getoption("--validate-test-config")

    for item in items:
        nodeid = item.nodeid
        if "[" in nodeid:
            nodeid = nodeid[nodeid.index("[") + 1 : -1]

        _collected_nodeids.add(nodeid)  # Track for final validation

        meta = ModelTestConfig(test_config.get(nodeid), arch)
        item._test_meta = meta  # attach for fixture access

        # Uncomment this to print info for each test.
        # print(f"DEBUG nodeid: {nodeid} meta.status: {meta.status}")

        if meta.status == ModelStatus.EXPECTED_PASSING:
            item.add_marker(pytest.mark.expected_passing)
        elif meta.status == ModelStatus.KNOWN_FAILURE_XFAIL:
            item.add_marker(pytest.mark.known_failure_xfail)
            item.add_marker(pytest.mark.xfail(strict=True, reason=meta.xfail_reason))
        elif meta.status == ModelStatus.NOT_SUPPORTED_SKIP:
            item.add_marker(pytest.mark.not_supported_skip)
        elif meta.status == ModelStatus.UNSPECIFIED:
            item.add_marker(pytest.mark.unspecified)

    # If validating config, clear all items so no tests run
    if validate_config:
        items.clear()


# Verify that the test_config.py file is valid.
def pytest_sessionfinish(session, exitstatus):
    if not session.config.getoption("--validate-test-config"):
        return  # Skip check unless explicitly requested

    print("\n" + "=" * 60)
    print("VALIDATING TEST CONFIGURATIONS")
    print("=" * 60 + "\n")

    declared_nodeids = set(test_config.keys())
    unknown = declared_nodeids - _collected_nodeids
    unlisted = _collected_nodeids - declared_nodeids
    print(
        f"Found {len(unknown)} unknown tests and {len(unlisted)} unlisted tests",
        flush=True,
    )

    # Unlisted tests are just warnings, for informational purposes.
    if unlisted:
        print("\nWARNING: The following tests are missing from test_config.py:")
        for test_name in sorted(unlisted):
            print(f"  - {test_name}")
    else:
        print("\nAll collected tests are properly defined in test_config.py")

    # Unknown tests are tests listed that no longer exist, treat as error.
    if unknown:
        msg = "test_config.py contains entries not found in collected tests."
        print(f"\nERROR: {msg}")
        for test_name in sorted(unknown):
            print(f"  - {test_name}")
            suggestion = difflib.get_close_matches(test_name, _collected_nodeids, n=1)
            if suggestion:
                print(f"    Did you mean: {suggestion[0]}?")
        print("\n" + "=" * 60)
        raise pytest.UsageError(msg)
    else:
        session.exitstatus = 0
