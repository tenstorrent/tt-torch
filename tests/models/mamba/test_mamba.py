# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/state-spaces/mamba-2.8b-hf

# from transformers import GenerationConfig
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
import torch
from third_party.tt_forge_models.mamba.pytorch.loader import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        model = self.loader.load_model(dtype_override=torch.bfloat16)

        # model.generate = lambda **kwargs: type(model).generate(
        #     model, **{**kwargs, "use_cache": False}
        # )

        self.tokenizer = self.loader.tokenizer

        return model

    def _load_inputs(self):
        input_ids = self.loader.load_inputs()  # ["input_ids"]
        # generation_config = GenerationConfig(max_new_tokens=10, use_cache=False)
        # arguments = {
        #     "input_ids": input_ids,
        #     "generation_config": generation_config,
        #     "use_cache": False,
        # }
        return input_ids


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
def test_mamba(record_property, variant_info, mode, op_by_op):

    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    variant, variant_config = variant_info
    loader = ModelLoader(variant=variant)
    model_info = loader.get_model_info(variant=variant)

    tester = ThisTester(
        model_info.name,
        mode,
        loader=loader,
        model_info=model_info,
        compiler_config=cc,
        record_property_handle=record_property,
        run_generate=False,
        required_pcc=0.95,
        assert_atol=False,
    )

    results = tester.test_model()

    if mode == "eval":
        logits = results.logits if hasattr(results, "logits") else results[0]
        token_ids = torch.argmax(logits, dim=-1)
        gen_text = tester.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        print("Generated text: ", gen_text)

    tester.finalize()
