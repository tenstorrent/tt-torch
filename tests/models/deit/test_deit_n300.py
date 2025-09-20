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
        return self.loader.load_inputs(dtype_override=torch.bfloat16, batch_size=16)

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
def test_deit(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()

    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.automatic_parallelization = True
    cc.mesh_shape = [1, 2]
    cc.dump_debug = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_info.name,
        mode,
        loader=loader,
        model_info=model_info,
        required_pcc=0.97,
        relative_atol=0.015,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=True,
        assert_atol=False,
        # FIXME fails with tt-experimental - https://github.com/tenstorrent/tt-torch/issues/1105
        backend="tt",
    )
    results = tester.test_model()

    if mode == "eval":
        logits = results.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_indices = logits.argmax(-1)
        for i, class_idx in enumerate(predicted_class_indices):
            print(
                f"Sample {i}: Predicted class: {tester.framework_model.config.id2label[class_idx.item()]}"
            )

    tester.finalize()
