# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512

import pytest
from tests.utils import ModelTester
import torch
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.segformer.semantic_segmentation.pytorch import (
    ModelLoader,
)


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs(dtype_override=torch.bfloat16)

    def set_inputs_train(self, inputs):
        inputs["pixel_values"] = inputs["pixel_values"].requires_grad_(True)
        return inputs

    def append_fake_loss_function(self, outputs):
        return torch.mean(outputs.logits)

    def get_results_train(self, model, inputs, outputs):
        return inputs["pixel_values"].grad


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
def test_segformer(record_property, mode, op_by_op, data_parallel_mode):
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

    tester = ThisTester(
        model_info.name,
        mode,
        loader=loader,
        relative_atol=0.01,
        compiler_config=cc,
        # TODO - Enable checking - https://github.com/tenstorrent/tt-torch/issues/527
        assert_atol=False,
        record_property_handle=record_property,
        model_group="red",
        data_parallel_mode=data_parallel_mode,
    )
    results = tester.test_model()

    def print_result(result):
        logits = result.logits  # shape (batch_size, num_labels, height/4, width/4)
        print(f"Logits: {logits}")

    if mode == "eval":
        ModelTester.print_outputs(results, data_parallel_mode, print_result)

    tester.finalize()
