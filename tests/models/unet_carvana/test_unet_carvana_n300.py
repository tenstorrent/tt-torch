# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/arief25ramadhan/carvana-unet-segmentation

import torch
import pytest

from tests.models.unet_carvana.carvana_unet_segmentation.model import UNET
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        model = UNET(in_channels=3, out_channels=1)
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        input_batch = torch.rand((4, 3, 224, 224))
        input_batch = input_batch.to(torch.bfloat16)
        return input_batch


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_unet_carvana(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "Unet-carvana"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.automatic_parallelization = True
    cc.mesh_shape = [1, 2]
    cc.dump_debug = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name, mode, compiler_config=cc, record_property_handle=record_property
    )
    tester.test_model()
    tester.finalize()
