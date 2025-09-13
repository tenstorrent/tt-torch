# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/facebook/deit-base-patch16-224

import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.deit.pytorch.loader import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs(dtype_override=torch.bfloat16)

    def set_inputs_train(self, inputs):
        inputs["pixel_values"].requires_grad_(True)
        return inputs

    def append_fake_loss_function(self, outputs):
        return torch.mean(outputs.logits)

    def get_results_train(self, model, inputs, outputs):
        return inputs["pixel_values"].grad


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(
            "train",
            marks=pytest.mark.compilation_xfail,
        ),
        "eval",
    ],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_deit(record_property, mode, op_by_op, data_parallel_mode):
    if mode == "train":
        pytest.skip()

    cc = CompilerConfig()
    cc.enable_consteval = True
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)

    # Enable Checking - https://github.com/tenstorrent/tt-torch/issues/1195
    assert_pcc = False if data_parallel_mode else True

    tester = ThisTester(
        model_info.name,
        mode,
        loader=loader,
        model_info=model_info,
        required_pcc=0.97,
        relative_atol=0.015,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=assert_pcc,
        assert_atol=False,
        data_parallel_mode=data_parallel_mode,
    )
    results = tester.test_model()

    def print_result(result):
        logits = result.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_indices = logits.argmax(-1)
        for i, class_idx in enumerate(predicted_class_indices):
            print(
                f"Sample {i}: Predicted class: {tester.framework_model.config.id2label[class_idx.item()]}"
            )

    if mode == "eval":
        ModelTester.print_outputs(results, data_parallel_mode, print_result)

    tester.finalize()
