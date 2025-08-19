# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import tt_torch
import pytest
import torch.nn.functional as F
# Load model directly
from transformers import Qwen3Config, AutoModel, AutoTokenizer
from torch import Tensor
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from tt_torch.tools.utils import (
    calculate_pcc,
)

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    # Extract tensor from BaseModelOutputWithPast if needed
    if hasattr(last_hidden_states, 'last_hidden_state'):
        last_hidden_states = last_hidden_states.last_hidden_state
    
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

sample_task = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
sample_queries = [
    "What is the capital of China?",
    "Explain gravity",
]
sample_documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

class ThisTester(ModelTester):
    def _load_model(self):
        model_name = "Qwen/Qwen3-Embedding-4B"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        config = Qwen3Config.from_pretrained(model_name)
        #config.num_hidden_layers =  # works on N150 upto 32 hidden layers. This model has 36 layers in total
        #print(f"Number of layers: {config.num_hidden_layers}")
        model = AutoModel.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16)
        # Set use_cache after loading
        model.config.use_cache = False
        print(f"Model parameter dtype: {next(model.parameters()).dtype}")
        model.eval()  

        return model

    def _load_inputs(self):
        queries = [
            get_detailed_instruct(sample_task, query)
            for query in sample_queries
        ]
        input_texts = queries + sample_documents
        max_length = 128
        # Tokenize the input texts
        inputs = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

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
def test_qwen3_embedding(record_property, mode, op_by_op):
    model_name = "Qwen/Qwen3-Embedding-4B"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        required_pcc=0.98,
        assert_pcc=True,
        assert_atol=False,
    )
    results = tester.test_model()

    
    # Helper function to decode output to human-readable text
    def decode_output(outputs, inputs=None):
        embeddings = last_token_pool(outputs, inputs["attention_mask"])

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity scores between queries and documents
        num_queries = len(sample_queries)
        scores = embeddings[:num_queries] @ embeddings[num_queries:].T

        return scores.tolist()

    decoded_output = decode_output(results, tester.inputs)

    print(f"Decoded output: {decoded_output}")

     
    tester.finalize()


def test_qwen3_embedding_demo(record_property):
    mode = "eval"
    op_by_op = None

    model_name = "Qwen/Qwen3-Embedding-4B"
    
    # Load model and inputs for CPU
    cpu_tokenizer = AutoTokenizer.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )
    config = Qwen3Config.from_pretrained(model_name)
    cpu_model = AutoModel.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16)
    # Set use_cache after loading
    cpu_model.config.use_cache = False
    cpu_model.eval()  

    queries = [
        get_detailed_instruct(sample_task, query)
        for query in sample_queries
    ]
    input_texts = queries + sample_documents
    max_length = 128
    # Tokenize the input texts
    cpu_inputs = cpu_tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    cpu_results = cpu_model(**cpu_inputs)


    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=True,
        assert_atol=False,
    )
    tt_results = tester.test_model()

    
    # Helper function to decode output to human-readable text
    def decode_output(outputs, inputs=None):
        embeddings = last_token_pool(outputs, inputs["attention_mask"])

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity scores between queries and documents
        num_queries = len(sample_queries)
        scores = embeddings[:num_queries] @ embeddings[num_queries:].T

        return scores.tolist()

    tt_decoded_output = decode_output(tt_results, tester.inputs)
    cpu_decoded_output = decode_output(cpu_results, tester.inputs)

    print(f"Decoded output TT: {tt_decoded_output}")
    print(f"Decoded output CPU: {cpu_decoded_output}")

    pcc = calculate_pcc(tt_decoded_output, cpu_decoded_output)
    print(f"PCC: {pcc}")

    assert pcc > 0.98, "PCC is less than 0.98"
     
    tester.finalize()