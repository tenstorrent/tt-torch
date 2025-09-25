# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.rmbg.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs(dtype_override=torch.bfloat16)


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_RMBG(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()

    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)
    model_name = model_info.name
    model_group = model_info.group.value

    cc = CompilerConfig()
    cc.enable_consteval = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        loader=loader,
        model_info=model_info,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
    )

    with torch.no_grad():
        results = tester.test_model()

    if mode == "eval":
        # Use loader's decode_output method to process and save result
        decoded_output = loader.decode_output(
            results, save_image=True, output_path="rmbg_output_image.png"
        )

        print(
            f"""
        model_name: {model_name}
        {decoded_output}
        """
        )

    tester.finalize()
