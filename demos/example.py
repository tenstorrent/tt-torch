# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest


@pytest.mark.parametrize(
    "a,b,expected",
    [
        pytest.param(1, 2, 3, marks=pytest.mark.expected_passing),
        pytest.param(
            2,
            2,
            5,
            marks=[
                pytest.mark.known_failure,
                pytest.mark.xfail(reason="Intentional bug"),
            ],
        ),
        pytest.param(3, 3, 6, marks=pytest.mark.expected_passing),
        pytest.param(
            4,
            5,
            10,
            marks=[pytest.mark.known_failure, pytest.mark.xfail(reason="Math is hard")],
        ),
    ],
)
def test_add(a, b, expected):
    assert a + b == expected
    # assert True
