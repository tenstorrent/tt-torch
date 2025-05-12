# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from tests.models.oft.src.oftnet import OftNet  # OftNet imports OFT


class ThisTester(ModelTester):
    def _load_model(self):
        # Load the OftNet model
        self.grid_res = 0.5
        model = OftNet(
            num_classes=1,
            frontend="resnet18",
            topdown_layers=8,
            grid_res=self.grid_res,
            grid_height=4.0,
        )
        return model

    def _load_inputs(self):
        # Create dummy inputs for the model
        batch_size = 32
        dummy_image = torch.randn(batch_size, 3, 224, 224)
        dummy_calib = torch.randn(batch_size, 3, 4)
        grid_size = (80.0, 80.0)  # width, depth
        grid_depth = 8
        grid_depth = int(grid_size[1] / self.grid_res)
        grid_width = int(grid_size[0] / self.grid_res)
        dummy_grid = torch.randn(batch_size, grid_depth, grid_width, 3)
        input = (dummy_image, dummy_calib, dummy_grid)
        return input


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_oft(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "OFT"
    model_group = "red"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.dump_info = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    #skip_full_eval_test(
    #    record_property,
    #    cc,
    #    model_name,
    #    bringup_status="FAILED_RUNTIME",
    #    reason="Out of Memory: Not enough space to allocate 2902982656 B DRAM buffer across 12 banks - https://github.com/tenstorrent/tt-torch/issues/727",
    #    model_group=model_group,
    #)

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        assert_atol=False,
        record_property_handle=record_property,
        model_group=model_group,
    )

    results = tester.test_model()
    tester.finalize()
