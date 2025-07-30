# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import difflib
import demos.test_config as test_config_module
from demos.utils import TestStatus, TestMeta


# 👇 Store items here for later access in terminal summary
collected_items = []

# 👇 Use dict of dicts now
test_config = test_config_module.test_config

# Define a test metadata fixture to get the test config for each test
@pytest.fixture
def test_metadata(request):
    return getattr(request.node, "_test_meta", TestMeta({}))


def pytest_addoption(parser):
    parser.addoption(
        "--list-nodeids",
        action="store_true",
        default=False,
        help="List test node IDs after collection",
    )


def pytest_collection_modifyitems(config, items):
    global collected_items
    collected_items = items

    for item in items:
        nodeid = item.nodeid.split("::")[-1]
        meta = TestMeta(test_config.get(nodeid))

        if meta.status == TestStatus.KNOWN_FAILURE:
            item.add_marker(pytest.mark.known_failure)
            item.add_marker(pytest.mark.xfail(strict=True, reason="Known failure"))
        elif meta.status == TestStatus.EXPECTED_PASSING:
            item.add_marker(pytest.mark.expected_passing)
        elif meta.status == TestStatus.TO_DEBUG:
            item.add_marker(pytest.mark.to_debug)

        # Attach TestMeta to the item so the test can access it
        item._test_meta = meta


def pytest_terminal_summary(terminalreporter):
    if terminalreporter.config.getoption("--list-nodeids"):
        print("\n📋 Collected test nodeids (use in test_config.py):")
        for item in collected_items:
            print(f'    "{item.nodeid.split("::")[-1]}",')


def pytest_sessionfinish(session, exitstatus):
    actual_nodeids = {item.nodeid.split("::")[-1] for item in collected_items}
    declared_nodeids = set(test_config.keys())

    unknown = declared_nodeids - actual_nodeids
    unlisted = actual_nodeids - declared_nodeids

    if unknown:
        print(
            "\nERROR: Unknown test entries in test_config.py (not found in collected tests):"
        )
        for test_name in sorted(unknown):
            print(f"  - {test_name}")
            suggestion = difflib.get_close_matches(test_name, actual_nodeids, n=1)
            if suggestion:
                print(f"    Did you mean: {suggestion[0]}?")
        session.exitstatus = 1  # fail

    if unlisted:
        print(
            "\nWARNING: The following tests are collected but missing from test_config.py:"
        )
        for test_name in sorted(unlisted):
            print(f"  - {test_name}")
        # Optional: treat as error
        # session.exitstatus = 1
