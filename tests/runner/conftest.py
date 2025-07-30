# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# tests/runner/conftest.py

import pytest
from tests.runner.test_config import test_config, ModelTestConfig, ModelStatus


def pytest_addoption(parser):
    parser.addoption(
        "--arch",
        action="store",
        default="blackhole",
        help="Target architecture (e.g., blackhole, wormhole)",
    )


@pytest.fixture
def test_metadata(request) -> ModelTestConfig:
    arch = request.config.getoption("--arch")
    nodeid = request.node.nodeid.split("::")[-1]
    return ModelTestConfig(test_config.get(nodeid), arch)


def pytest_collection_modifyitems(config, items):
    arch = config.getoption("--arch")

    for item in items:
        nodeid = item.nodeid.split("::")[-1]
        meta = ModelTestConfig(test_config.get(nodeid), arch)
        item._test_meta = meta  # attach for fixture access

        # Add pytest markers so `-m` filtering works
        if meta.status == ModelStatus.EXPECTED_PASSING:
            item.add_marker(pytest.mark.expected_passing)
        elif meta.status == ModelStatus.KNOWN_FAILURE:
            item.add_marker(pytest.mark.known_failure)
        elif meta.status == ModelStatus.TO_DEBUG:
            item.add_marker(pytest.mark.to_debug)
