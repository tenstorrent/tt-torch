# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.bert.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        # We need to call ModelLoader.load_model once to initialize its tokenizer
        # which is a class attribute (ModelLoader.tokenizer).
        # We also need to store the model instance.
        self.model_loader_instance = ModelLoader.load_model(
            dtype_override=torch.bfloat16
        )
        return self.model_loader_instance

    def _load_inputs(self):
        """Load and return sample inputs for the BERT model with a custom batch size.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        # Define your batch of questions and contexts here
        batch_questions = [
            "What discipline did Winkelmann create?",
            "What discipline did Winkelmann create?",
            "What discipline did Winkelmann create?",
            "What discipline did Winkelmann create?",
        ]
        batch_contexts = [
            'Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. "The prophet and founding hero of modern archaeology", Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art. ',
            'Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. "The prophet and founding hero of modern archaeology", Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art. ',
            'Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. "The prophet and founding hero of modern archaeology", Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art. ',
            'Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. "The prophet and founding hero of modern archaeology", Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art. ',
        ]
        max_length = 256  # You can also customize this if needed

        # Ensure ModelLoader's tokenizer is initialized.
        # Calling ModelLoader.load_model() again would re-initialize it,
        # but if we already loaded the model, its tokenizer is available.
        # Access the tokenizer from the class attribute of ModelLoader.
        if not hasattr(ModelLoader, "tokenizer"):
            ModelLoader.load_model()  # This ensures ModelLoader.tokenizer is set

        # Create tokenized inputs for the batch
        inputs = ModelLoader.tokenizer(  # Use ModelLoader.tokenizer directly
            batch_questions,
            batch_contexts,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        # Store the original questions and contexts for decoding later
        self.batch_questions = batch_questions
        self.batch_contexts = batch_contexts

        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_bert(record_property, mode, op_by_op):
    model_name = "BERT"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.automatic_parallelization = True
    cc.mesh_shape = [1, 2]
    cc.dump_debug = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        relative_atol=0.012,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=True,
        assert_atol=False,
    )
    results = tester.test_model()

    if mode == "eval":
        # Ensure that ModelLoader.decode_output can handle batch outputs
        # and it needs access to the original inputs for token IDs.
        # The ModelLoader.decode_output expects `inputs` and returns a list.
        answers = ModelLoader.decode_output(results, tester.inputs)

        print(f"\nModel Name: {model_name}\n")
        for i, answer in enumerate(answers):
            print(f"--- Question {i+1} ---")
            print(f"  Question: {tester.batch_questions[i]}")
            print(
                f"  Context: {tester.batch_contexts[i][:100]}..."
            )  # Print first 100 chars of context
            print(f"  Answer: {answer}\n")

    tester.finalize()
