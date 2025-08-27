# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.stable_diffusion_3_5.vae.pytorch import (
    ModelLoader,
)


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs(dty)


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
def test_stable_diffusion_vae(record_property, variant, variant_config, mode, op_by_op):
    loader = ModelLoader(variant=variant)
    model_info = loader.get_model_info(variant=variant)
    model_name = model_info.name

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True  # will run into OOM error faster if this is disabled
    cc.dump_debug = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    if variant.value in ["medium-encoder"]:
        reason = "Out of Memory: Not enough space to allocate 84213760 B L1 buffer across 64 banks, where each bank needs to store 1315840 B"
    elif variant.value in ["medium-decoder"]:
        reason = "Out of Memory: Not enough space to allocate 71860224 B L1 buffer across 64 banks, where each bank needs to store 1122816 B"
    else:
        reason = (
            "medium-vae encounters OOM errors, so skipping large-vae full eval test"
        )

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_RUNTIME",
        reason=reason,
    )

    tester = ThisTester(
        model_name,
        mode,
        loader=loader,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=False,
        assert_pcc=True,
        model_group=model_info.group,
    )
    breakpoint()
    results = tester.test_model()
    tester.finalize()
