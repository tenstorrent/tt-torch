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
def test_add(a, b, expected):
    assert a + b == expected
