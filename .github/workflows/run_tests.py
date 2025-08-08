# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys

if __name__ == "__main__":
    with open(".pytest_tests_to_run", "r") as fd:
        test_list = [line.strip() for line in fd.readlines()]
    print(f"KCM Going to run pytest on tests: {test_list}", flush=True)
    sys.exit(pytest.main(test_list + sys.argv[1:]))
