# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.resnet.resnet_50_tv.pytorch import ModelLoader


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
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_resnet(record_property, mode, op_by_op, data_parallel_mode):
    if mode == "train":
        pytest.skip()

    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)
    model_name = model_info.name
    model_group = model_info.group.value

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    assert_pcc = True
    required_pcc = 0.97

    tester = ThisTester(
        model_name,
        mode,
        loader=loader,
        required_atol=0.03,
        required_pcc=required_pcc,
        compiler_config=cc,
        assert_pcc=assert_pcc,
        assert_atol=False,
        record_property_handle=record_property,
        data_parallel_mode=data_parallel_mode,
        model_group=model_group,
    )

    results = tester.test_model()

    def print_result(result):
        _, indices = torch.topk(result, 5)
        print(f"Top 5 predictions: {indices[0].tolist()}")

    if mode == "eval":
        ModelTester.print_outputs(results, data_parallel_mode, print_result)

        # Use loader's decode_output method for additional info
        decoded_output = loader.decode_output(results)
        print(decoded_output)

    tester.finalize()


# Empty property record_property
def empty_record_property(a, b):
    pass


# Run pytorch implementation
if __name__ == "__main__":
    test_resnet(
        empty_record_property, ModelLoader.ModelVariant.TV, {}, "eval", None, False
    )
