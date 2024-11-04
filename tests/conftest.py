# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.fixture(autouse=True)
def run_around_tests():
    torch.manual_seed(0)
    yield
    torch._dynamo.reset()
