# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/arief25ramadhan/carvana-unet-segmentation

import os
import subprocess
import sys
from pathlib import Path
import tempfile
import torch
import pytest

from tests.models.unet_carvana.carvana_unet_segmentation.model import UNET
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth


class ThisTester(ModelTester):
    def _load_model(self):
        model = UNET(in_channels=3, out_channels=1)
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        input_batch = torch.rand((1, 3, 224, 224))
        input_batch = input_batch.to(torch.bfloat16)
        return input_batch


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize("op_by_op", [True, False], ids=["op_by_op", "full"])
def test_unet_carvana(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "Unet-carvana"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    tester = ThisTester(
        model_name, mode, compiler_config=cc, record_property_handle=record_property
    )
    tester.test_model()
    tester.finalize()
