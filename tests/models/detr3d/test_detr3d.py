# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.detr3d.pytorch.loader import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        model_dict = dict(model_info_list)
        model_path = model_dict[self.model_name]
        return self.loader.load_model(variant_name=model_path)

    def _load_inputs(self):
        model_dict = dict(model_info_list)
        model_path = model_dict[self.model_name]
        return self.loader.load_inputs(variant_name=model_path)


model_info_list = [
    ("detr3d", "resnet101"),
]


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_info",
    model_info_list,
    ids=[model_info[0] for model_info in model_info_list],
)
@pytest.mark.parametrize(
    "op_by_op",
    # [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    # ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
    [None],
    ids=["full"],
)
def test_detr3d(record_property, model_info, mode, op_by_op):
    model_name, _ = model_info

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    # TODO - update this test once centernet ModelLoader is updated to properly support variants
    # for now it contains a few hacks.
    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)

    skip_full_eval_test(
        record_property,
        cc,
        model_info.name,
        bringup_status="FAILED_RUNTIME",
        reason="Out of Memory: Not enough space to allocate 285081600 B L1 buffer across 64 banks, where each bank needs to store 4454400 B",
        model_group=model_info.group,
    )

    # TODO Enable PCC/ATOL/Checking - https://github.com/tenstorrent/tt-torch/issues/976
    tester = ThisTester(
        model_name,
        mode,
        loader=loader,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_info.group,
    )

    results = tester.test_model()

    if mode == "eval":
        # Results
        print(results)

    tester.finalize()
