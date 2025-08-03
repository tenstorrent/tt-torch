# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.runner.test_config import test_config
from tests.runner.test_utils import ModelTestConfig, ModelStatus
import difflib

# Global set to track collected test node IDs
_collected_nodeids = set()


def pytest_addoption(parser):
    parser.addoption(
        "--arch",
        action="store",
        default="blackhole",
        help="Target architecture (e.g., blackhole, wormhole)",
    )
    parser.addoption(
        "--validate-test-config",
        action="store_true",
        default=False,
        help="Fail if test_config.py and collected test IDs are out of sync",
    )


@pytest.fixture(autouse=True)
def log_test_name(request):
    print(f"\nRunning {request.node.nodeid}", flush=True)


@pytest.fixture
def test_metadata(request) -> ModelTestConfig:
    meta = getattr(request.node, "_test_meta", None)
    assert meta is not None, f"No ModelTestConfig attached for {request.node.nodeid}"
    return meta


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
        print(f"DEBUG nodeid: {nodeid} meta.status: {meta.status}")

        if meta.status == ModelStatus.EXPECTED_PASSING:
            item.add_marker(pytest.mark.expected_passing)
        elif meta.status == ModelStatus.KNOWN_FAILURE_XFAIL:
            item.add_marker(pytest.mark.known_failure_xfail)
            # FIXME - Make xfail reason able to be provided in test_config
            item.add_marker(pytest.mark.xfail(strict=True, reason="Known failure"))
        elif meta.status == ModelStatus.NOT_SUPPORTED_SKIP:
            item.add_marker(pytest.mark.not_supported_skip)
        elif meta.status == ModelStatus.UNSPECIFIED:
            item.add_marker(pytest.mark.unspecified)
            item.add_marker(
                pytest.mark.xfail(
                    strict=False, reason="Status unspecified - awaiting triage"
                )
            )

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

    if unlisted:
        print("\nWARNING: The following tests are missing from test_config.py:")
        for test_name in sorted(unlisted):
            print(f"  - {test_name}")
    else:
        print("\nAll collected tests are properly defined in test_config.py")

    if unknown:
        print("\nERROR: test_config.py contains entries not found in collected tests:")
        for test_name in sorted(unknown):
            print(f"  - {test_name}")
            suggestion = difflib.get_close_matches(test_name, _collected_nodeids, n=1)
            if suggestion:
                print(f"    Did you mean: {suggestion[0]}?")
        print("\n" + "=" * 60)

    if not unknown and not unlisted:
        print("Test configuration validation PASSED - all entries match!")
    elif unknown:
        raise pytest.UsageError(
            "test_config.py contains entries not found in collected tests."
        )
