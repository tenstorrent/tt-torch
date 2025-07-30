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

    print(
        f"[INFO] status={test_metadata.status}, "
        f"batch_size={test_metadata.batch_size}, "
        f"pcc={test_metadata.pcc}"
    )

    # Illustrative example for accessing enum
    from conftest import TestStatus

    if test_metadata.status == TestStatus.KNOWN_FAILURE:
        print(f"[INFO] Known failure")

    assert a + b == expected
