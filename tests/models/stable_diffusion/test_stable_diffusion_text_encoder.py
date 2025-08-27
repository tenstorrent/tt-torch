# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.stable_diffusion_3_5.text_encoder.pytorch import (
    ModelLoader,
)


class TextEncoderTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs()


# Print available variants for reference
available_variants = ModelLoader.query_available_variants()
print("Available variants: ", [str(k) for k in available_variants.keys()])


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "variant, variant_config",
    available_variants.items(),
    ids=[str(k) for k in available_variants.keys()],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_stable_diffusion_text_encoder(
    record_property, variant, variant_config, mode, op_by_op
):
    loader = ModelLoader(variant=variant)
    model_info = loader.get_model_info(variant=variant)
    model_name = model_info.name

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    assert_pcc = (
        True
        if (
            model_name == "SD3.5-medium-text-encoder-3"
            or model_name == "SD3.5-large-text-encoder-3"
        )
        else False
    )

    tester = TextEncoderTester(
        model_name,
        mode,
        loader=loader,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=False,
        assert_pcc=assert_pcc,
        model_group=model_info.group,
    )
    results = tester.test_model()
    tester.finalize()
