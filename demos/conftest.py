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
