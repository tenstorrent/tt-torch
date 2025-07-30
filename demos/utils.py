# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# demos/utils.py

from enum import Enum


class TestMeta:
    def __init__(self, data):
        self.data = data or {}
        self.status = self._parse_status()

    def _parse_status(self):
        status = self.data.get("status", TestStatus.EXPECTED_PASSING)
        if not isinstance(status, TestStatus):
            raise ValueError(f"Expected TestStatus enum, got: {status}")
        return status

    @property
    def batch_size(self) -> int:
        return self.data.get("batch_size", 1)

    @property
    def pcc(self) -> float:
        return self.data.get("pcc", 0.98)


class TestStatus(Enum):
    EXPECTED_PASSING = "expected_passing"
    KNOWN_FAILURE = "known_failure"
    TO_DEBUG = "to_debug"
