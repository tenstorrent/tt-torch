# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 2, 3),
        (2, 2, 5),  # This is a known failure
        (3, 3, 6),
        (3, 5, 8),
        (4, 5, 10),  # Another known failure
    ],
)
def test_add(a, b, expected, test_metadata):

    status = test_metadata.status
    batch_size = test_metadata.batch_size
    pcc = test_metadata.pcc
    print(f"[INFO] status={status}, batch_size={batch_size}, pcc={pcc}")

    assert a + b == expected
