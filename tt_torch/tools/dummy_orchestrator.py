# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# run as
# python3 tt_torch/tools/dummy_orchestrator.py
import perf
import pytest


def main():
    cvar = perf.Perf()
    cvar.setup_tracy_server()
    pytest.main(
        ["-svv", "tests/models/mnist/test_mnist.py::test_mnist_train[full-eval]"]
    )
    # pytest.main(["-svv", "tests/models/mnist/test_mnist.py::test_mnist_train[op_by_op-eval]"])


if __name__ == "__main__":
    main()
