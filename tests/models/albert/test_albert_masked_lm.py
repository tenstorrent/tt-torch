# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/albert#transformers.AlbertForMaskedLM

from transformers import AutoTokenizer, AlbertForMaskedLM
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        return AlbertForMaskedLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )

    def _load_inputs(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
        self.model_name, torch_dtype=torch.bfloat16
        )

        questions = [
            "The capital of France is [MASK].",
            "[MASK] is the largest country in the world by area.",
            "The Great Wall of [MASK] is a famous landmark.",
            "Mount [MASK] is the highest mountain in the world.",
            "Water freezes at 0 degrees [MASK].",
            "The Earth orbits around the [MASK].",
            "[MASK] is the process by which plants make food from sunlight.",
            "The chemical symbol for oxygen is [MASK].",
            "[MASK] was the first President of the United States.",
            "The [MASK] Empire was known for its gladiators and coliseums.",
            "The Declaration of Independence was signed in [MASK].",
            "[MASK] was the leader of Nazi Germany during World War II.",
            "[MASK] is the founder of Microsoft.",
            "The [MASK] is a device used to input text into a computer.",
            "A smartphone typically has a [MASK] screen.",
            "The [MASK] is used to browse websites.",
            "A triangle has [MASK] sides.",
            "People usually wear [MASK] on their feet.",
            "You eat soup with a [MASK].",
            "A dog is a type of [MASK].",
            "Pizza is typically topped with cheese and [MASK].",
            "You use a [MASK] to drink a milkshake.",
            "[MASK] is a popular fruit that's yellow and curved.",
            "A sandwich usually contains bread and [MASK].",
            "[MASK] is known for playing Iron Man in the Marvel movies.",
            "The wizarding school in Harry Potter is called [MASK].",
            "[MASK] is the superhero alter ego of Bruce Wayne.",
            "The television show about a group of friends in New York is called [MASK].",
            "If it’s raining, you should bring an [MASK].",
            "You sleep on a [MASK] at night.",
            "You brush your teeth with a [MASK].",
            "The opposite of hot is [MASK]."
        ]

        self.inputs = self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
        return self.inputs

    def set_inputs_train(self, inputs):
        return inputs

    def append_fake_loss_function(self, outputs):
        return torch.mean(outputs.logits)

    # TODO: inputs has no grad, how to get it?
    # def get_results_train(self, model, inputs, outputs):
    #     return


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "albert/albert-base-v2",
        "albert/albert-large-v2",
        "albert/albert-xlarge-v2",
        "albert/albert-xxlarge-v2",
    ],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_albert_masked_lm(record_property, model_name, mode, op_by_op):

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.dump_info = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()

    if mode == "eval":
        # retrieve index of [MASK]
        logits = results.logits
        mask_token_index = (tester.inputs.input_ids == tester.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_tokens = tester.tokenizer.decode(predicted_token_id)

        print(f"Model: {model_name} | Input: {tester.text} | Mask: {predicted_tokens}")

    tester.finalize()
