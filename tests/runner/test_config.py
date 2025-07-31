# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelStatus


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
