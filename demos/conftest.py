# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import demos.test_config as test_config  # adjust path if needed


# 👇 Store items here for later access in terminal summary
collected_items = []


def pytest_addoption(parser):
    parser.addoption(
        "--list-nodeids",
        action="store_true",
        default=False,
        help="List test node IDs after collection",
    )


def pytest_collection_modifyitems(config, items):
    global collected_items
    collected_items = items  # store for later

    for item in items:
        nodeid = item.nodeid.split("::")[-1]
        if nodeid in test_config.known_failures:
            item.add_marker(pytest.mark.known_failure)
            item.add_marker(pytest.mark.xfail(strict=True, reason="Known failure"))
        elif nodeid in test_config.expected_passing:
            item.add_marker(pytest.mark.expected_passing)


def pytest_terminal_summary(terminalreporter):
    if terminalreporter.config.getoption("--list-nodeids"):
        print("\n📋 Collected test nodeids (use in test_config.py):")
        for item in collected_items:
            print(f'    "{item.nodeid.split("::")[-1]}",')


# Validate config file to report on unexpected or new model tests
def pytest_sessionfinish(session, exitstatus):
    actual_nodeids = {item.nodeid.split("::")[-1] for item in collected_items}
    declared = set(test_config.expected_passing) | set(test_config.known_failures)

    unknown = declared - actual_nodeids
    unlisted = actual_nodeids - declared

    if unknown:
        print(
            "\nERROR: Unknown test entries in test_config.py (not found in collected tests):"
        )
        for test_name in sorted(unknown):
            print(f"  - {test_name}")
        session.exitstatus = 1  # fail

    if unlisted:
        print(
            "\nWARNING: The following tests are collected but missing from test_config.py:"
        )
        for test_name in sorted(unlisted):
            print(f"  - {test_name}")
        # Optional: uncomment the next line to treat unlisted as an error
        # session.exitstatus = 1
