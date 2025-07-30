# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# test_config.py

expected_passing = {
    "test_add[1-2-3]",
    "test_add[3-3-6]",
    "test_add[5-3-6]",
}

known_failures = {
    "test_add[2-2-5]",
    "test_add[4-5-10]",
}
