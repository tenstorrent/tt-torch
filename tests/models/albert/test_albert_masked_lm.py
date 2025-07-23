# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/albert#transformers.AlbertForMaskedLM

import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, ModelMetadata
from tt_torch.tools.utils import construct_metadata_from_variants
from third_party.tt_forge_models.albert.masked_lm.pytorch import ModelLoader


class ThisTester(ModelTester):
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


OVERRIDE_VARIANTS = {
    "albert-base-v2": ModelMetadata(
        variant_name="albert-base-v2",
        assert_atol=False,
    ),
}

variant_metadata_list, variant_ids = construct_metadata_from_variants(
    ModelLoader, OVERRIDE_VARIANTS
)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
# variant_info is a ModelMetadata object with ModelInfo, variant_config
# and other tt-forge-models FE agnostic info embedded in it
@pytest.mark.parametrize(
    "variant_info",
    variant_metadata_list,
    ids=variant_ids,
)
@pytest.mark.parametrize(
    "execute_mode",
    [CompileDepth.EXECUTE_OP_BY_OP, CompileDepth.EXECUTE],
    ids=["op_by_op", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_albert_masked_lm(
    record_property, variant_info, mode, execute_mode, data_parallel_mode
):
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    cc.op_by_op_backend = variant_info.op_by_op_backend
    if execute_mode == CompileDepth.EXECUTE_OP_BY_OP:
        cc.compile_depth = execute_mode
    else:
        cc.compile_depth = variant_info.compile_depth

    variant = variant_info.variant_name
    variant_config = variant_info.variant_config

    model_info = variant_info.loader.get_model_info(variant=variant)
    model_name = model_info.name

    variant_info.assert_pcc = True
    variant_info.required_pcc = 0.98

    tester = ThisTester(
        model_name,
        mode,
        loader=variant_info.loader,
        required_pcc=variant_info.required_pcc,
        assert_pcc=variant_info.assert_pcc,
        assert_atol=variant_info.assert_atol,
        compiler_config=cc,
        record_property_handle=record_property,
        data_parallel_mode=data_parallel_mode,
    )
    results = tester.test_model()

    def print_result(result):
        predicted_tokens = variant_info.loader.decode_output(result, tester.inputs)
        print(f"Model: {model_name} | Input: {tester.text} | Mask: {predicted_tokens}")

    if mode == "eval":
        ModelTester.print_outputs(results, data_parallel_mode, print_result)

    tester.finalize()
