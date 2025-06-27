# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from tests.utils import OnnxModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.centernet.onnx import ModelLoader


class ThisTester(OnnxModelTester):
    def _load_model(self):
        model_dict = dict(model_info_list)
        model_path = model_dict[self.model_name]
        return self.loader.load_model(variant_name=model_path)

    def _load_torch_inputs(self):
        model_dict = dict(model_info_list)
        model_path = model_dict[self.model_name]
        return self.loader.load_inputs(variant_name=model_path)


model_info_list = [
    ("centernet-od-dla1x", "dla1x_od"),
    ("centernet-od-dla2x", "dla2x_od"),
    ("centernet-od-resdcn18", "resdcn18_od"),
    ("centernet-od-resdcn101", "resdcn101_od"),
    ("centernet-od-hg", "hg_od"),
    ("centernet-hpe-dla1x", "dla1x_hpe"),
    ("centernet-hpe-dla3x", "dla3x_hpe"),
    ("centernet-hpe-hg3x", "hg3x_hpe"),
    ("centernet-3d_bb-ddd_3dop", "ddd_3dop_3d_bb"),
    ("centernet-3d_bb-ddd_sub", "ddd_sub_3d_bb"),
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
    [OpByOpBackend.STABLEHLO, None],
    ids=["op_by_op_stablehlo", "full"],
)
def test_centernet_onnx(record_property, model_info, mode, op_by_op):
    model_name, _ = model_info
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    if op_by_op is not None:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        cc.op_by_op_backend = op_by_op

    # TODO - update this test once centernet ModelLoader is updated to properly support variants
    # for now it contains a few hacks.
    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_RUNTIME",
        reason="'ttir.conv_transpose2d' op Number of input channels from input tensor must match the first dimension of the weight tensor. Got 256 input channels and 1 in the weight tensor.",
        model_group=model_info.group,
        model_name_filter=[
            "centernet-od-dla1x",
            "centernet-od-dla2x",
            "centernet-od-hg",
            "centernet-hpe-dla1x",
            "centernet-hpe-dla3x",
            "centernet-hpe-hg3x",
            "centernet-3d_bb-ddd_3dop",
            "centernet-3d_bb-ddd_sub",
        ],
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
