# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class ModelStatus(Enum):
    # Passing tests
    EXPECTED_PASSING = "expected_passing"
    # Known failures that should be xfailed
    KNOWN_FAILURE_XFAIL = "known_failure_xfail"
    # Not supported on this architecture or low priority
    NOT_SUPPORTED_SKIP = "not_supported_skip"
    # New model, awaiting triage
    UNSPECIFIED = "unspecified"


class ModelTestConfig:
    def __init__(self, data: dict, arch: str):
        self.data = data or {}
        self.arch = arch
        self.status = self._resolve("status", default=ModelStatus.UNSPECIFIED)
        self.pcc = self._resolve("pcc", default=0.98)
        self.batch_size = self._resolve("batch_size", default=1)
        self.assert_pcc = self._resolve("assert_pcc", default=True)
        self.assert_atol = self._resolve("assert_atol", default=False)
        self.relative_atol = self._resolve("relative_atol", default=None)

    def _resolve(self, key, default=None):
        overrides = self.data.get("arch_overrides", {})
        if self.arch in overrides and key in overrides[self.arch]:
            return overrides[self.arch][key]
        return self.data.get(key, default)


# TODO - Add a step that verifies configs are valid.
test_config = {
    "bloom/pytorch-full-eval": {
        "pcc": False,
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
    },
    "xglm/pytorch-xglm-564M-full-eval": {
        "pcc": False,
    },
    "gpt_neo/pytorch-full-eval": {"pcc": 0.98},
    "vovnet/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "pcc": False,
    },
    "phi3/pytorch-mini_4k_instruct-full-eval": {
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "batch_size": 2,
        "arch_overrides": {
            "wormhole": {
                "status": ModelStatus.EXPECTED_PASSING,
                "pcc": 0.99,
            }
        },
    },
}
