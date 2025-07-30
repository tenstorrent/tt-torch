# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# test_config.py

from demos.utils import TestStatus

test_config = {
    "test_add[1-2-3]": {
        "status": TestStatus.EXPECTED_PASSING,
        "batch_size": 8,
        "pcc": 0.98,
    },
    "test_add[3-3-6]": {
        "status": TestStatus.EXPECTED_PASSING,
        "batch_size": 16,
        "pcc": 0.97,
    },
    "test_add[5-3-6]": {
        "status": TestStatus.EXPECTED_PASSING,
        "batch_size": 4,
        "pcc": 0.95,
    },
    "test_add[2-2-5]": {
        "status": TestStatus.KNOWN_FAILURE,
        "batch_size": 2,
        "pcc": 0.90,
    },
    "test_add[4-5-10]": {
        "status": TestStatus.KNOWN_FAILURE,
        "batch_size": 1,
        "pcc": 0.88,
    },
}
