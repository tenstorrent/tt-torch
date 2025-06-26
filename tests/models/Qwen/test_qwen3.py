# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/Qwen/Qwen3-Embedding-8B

from transformers import AutoTokenizer, AutoModel
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        return model.eval()

    def _load_inputs(self):
        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f"Instruct: {task_description}\nQuery:{query}"

        task = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        queries = [
            get_detailed_instruct(task, "What is the capital of China?"),
            get_detailed_instruct(task, "Explain gravity"),
        ]
        documents = [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        ]
        input_texts = queries + documents
        # Tokenize the input texts
        input_tokens = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = [input_tokens["input_ids"]]
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen3-Embedding-0.6B",
    ],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.TORCH, None],
    ids=["op_by_op_torch", "full"],
)
def test_qwen3(record_property, model_name, mode, op_by_op):
    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=False,
        assert_pcc=False,
    )

    results = tester.test_model()

    print(f"Result: {results}")

    tester.finalize()
