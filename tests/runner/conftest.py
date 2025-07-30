# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# tests/runner/conftest.py

import pytest
from tests.runner.test_config import test_config, TestMeta


def pytest_addoption(parser):
    parser.addoption(
        "--arch",
        action="store",
        default="blackhole",
        help="Target architecture (e.g., blackhole, wormhole)",
    )


@pytest.fixture
def test_metadata(request) -> TestMeta:
    arch = request.config.getoption("--arch")
    nodeid = request.node.nodeid.split("::")[-1]
    return TestMeta(test_config.get(nodeid), arch)
