# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    ModelMetadata,
)
from tt_torch.tools.utils import construct_metadata_from_variants
from third_party.tt_forge_models.oft.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model()

    def _load_inputs(self):
        return self.loader.load_inputs()


OVERRIDE_VARIANTS = {
    "base": {
        ModelMetadata(
            variant_name="base",
            assert_atol=False,
        )
    }
}

variant_metadata_list, variant_ids = construct_metadata_from_variants(
    ModelLoader, OVERRIDE_VARIANTS
)


@pytest.mark.parametrize(
    "variant_info",
    variant_metadata_list,
    ids=variant_ids,
)
@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
def test_oft(record_property, mode, execute_mode, variant_info):
    if mode == "train":
        pytest.skip()

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    cc.op_by_op_backend = variant_info.op_by_op_backend
    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
        cc.compile_depth = execute_mode
    else:
        cc.compile_depth = variant_info.compile_depth

    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)

    skip_full_eval_test(
        record_property,
        cc,
        model_info.name,
        bringup_status="FAILED_RUNTIME",
        reason="Out of Memory: Not enough space to allocate 2902982656 B DRAM buffer across 12 banks - https://github.com/tenstorrent/tt-torch/issues/727",
        model_group=model_info.group,
    )

    tester = ThisTester(
        model_info.name,  # name of model
        mode,
        loader=loader,
        model_info=model_info,
        compiler_config=cc,
        assert_atol=variant_info.assert_atol,
        record_property_handle=record_property,
    )

    results = tester.test_model()
    tester.finalize()
