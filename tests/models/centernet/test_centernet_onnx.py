# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from tests.utils import OnnxModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.centernet.pytorch import ModelLoader


class ThisTester(OnnxModelTester):
    def _load_model(self):
        """
        The model is from https://github.com/xingyizhou/CenterNet
        """
        # Model
        model_dict = dict(model_info_list)
        model_path = model_dict[self.model_name]
        return ModelLoader.load_model(variant_name=model_path)

    def _load_torch_inputs(self):
        # Images
        model_dict = dict(model_info_list)
        model_path = model_dict[self.model_name]
        return ModelLoader.load_inputs(variant_name=model_path)


model_info_list = [
    ("centernet-od-dla1x", "dla1x_od"),
    ("centernet-od-dla2x", "dla2x_od"),
    ("centernet-od-resdcn18", "resdcn18_od"),
    ("centernet-od-resdcn101", "resdcn101_od"),
    ("centernet-od-hg", "hg_ob"),
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
    model_group = "red"
    model_name, _ = model_info
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    if op_by_op is not None:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        cc.op_by_op_backend = op_by_op

    skip_full_eval_test(
        record_property,
        cc,
        model_name,
        bringup_status="FAILED_RUNTIME",
        reason="'ttir.conv_transpose2d' op Number of input channels from input tensor must match the first dimension of the weight tensor. Got 256 input channels and 1 in the weight tensor.",
        model_group=model_group,
    )

    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=True,
        assert_atol=True,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
    )
    results = tester.test_model()

    if mode == "eval":
        # Results
        print(results)

    tester.finalize()
