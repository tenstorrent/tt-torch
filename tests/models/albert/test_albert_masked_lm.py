# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/albert#transformers.AlbertForMaskedLM

import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.albert.masked_lm.pytorch import ModelLoader


class ThisTester(ModelTester):
    def __init__(self, model_name, mode, variant=None, **kwargs):
        self.variant = variant
        self.loader = ModelLoader(variant=variant)
        super().__init__(model_name, mode, **kwargs)

    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        self.inputs = self.loader.load_inputs(dtype_override=torch.bfloat16)
        self.text = ModelLoader.sample_text
        self.tokenizer = self.loader.tokenizer
        return self.inputs

    def set_inputs_train(self, inputs):
        return inputs

    def append_fake_loss_function(self, outputs):
        return torch.mean(outputs.logits)

    # TODO: inputs has no grad, how to get it?
    # def get_results_train(self, model, inputs, outputs):
    #     return


# Print available variants for reference
available_variants = ModelLoader.query_available_variants()
print("Available variants:", available_variants)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "variant_info",
    available_variants.items(),
    ids=list(available_variants.keys()),
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_albert_masked_lm(
    record_property, variant_info, mode, op_by_op, data_parallel_mode
):
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    variant, variant_config = variant_info
    model_name = f"albert/{variant}-masked_lm"
    print(f"Testing model_name: {model_name} variant: {variant}", flush=True)

    required_pcc = 0.98 if "xxlarge" in variant else 0.99

    tester = ThisTester(
        model_name,
        mode,
        variant=variant,
        required_pcc=required_pcc,
        assert_pcc=True,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        data_parallel_mode=data_parallel_mode,
    )
    results = tester.test_model()

    def print_result(result):
        predicted_tokens = tester.loader.decode_output(result, tester.inputs)
        print(f"Model: {model_name} | Input: {tester.text} | Mask: {predicted_tokens}")

    if mode == "eval":
        ModelTester.print_outputs(results, data_parallel_mode, print_result)

    tester.finalize()
