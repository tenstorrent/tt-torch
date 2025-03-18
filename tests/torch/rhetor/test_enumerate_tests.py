# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from tt_torch.tools.crashsafe_utils import *


def test_enumerate_all_model_tests():
    test_list = enumerate_all_tests()

    # prune list to filter out full eval
    test_list = [test for test in test_list if ("full" in test and "eval" in test)]
    # for test in test_list:
    #     print(test)

    matrix = generate_test_matrix(test_list)
    for m in matrix:
        print(repr(m) + ",\n")
    assert len(test_list) > 0
