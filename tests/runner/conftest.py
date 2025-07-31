# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.runner.test_config import test_config
from tests.runner.test_utils import ModelTestConfig, ModelStatus


def pytest_addoption(parser):
    parser.addoption(
        "--arch",
        action="store",
        default="blackhole",
        help="Target architecture (e.g., blackhole, wormhole)",
    )


@pytest.fixture
def test_metadata(request) -> ModelTestConfig:
    meta = getattr(request.node, "_test_meta", None)
    assert meta is not None, f"No ModelTestConfig attached for {request.node.nodeid}"
    return meta


def pytest_collection_modifyitems(config, items):
    arch = config.getoption("--arch")

    for item in items:
        # Extract just the test ID inside brackets, used as key in test_config.
        nodeid = item.nodeid
        if "[" in nodeid:
            nodeid = nodeid[nodeid.index("[") + 1 : -1]

        meta = ModelTestConfig(test_config.get(nodeid), arch)
        item._test_meta = meta  # attach for fixture access

        # Uncomment this to print info for each test.
        # print(f"KCM nodeid: {nodeid} meta.status: {meta.status}")

        # Add markers so they can be filtered via -m and behave properly during run
        if meta.status == ModelStatus.EXPECTED_PASSING:
            item.add_marker(pytest.mark.expected_passing)
        elif meta.status == ModelStatus.KNOWN_FAILURE_XFAIL:
            item.add_marker(pytest.mark.known_failure_xfail)
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
