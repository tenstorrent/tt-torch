# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaModel
import torch_xla
import torch_xla.core.xla_model as xm

from tt_torch.tools.utils import (
    calculate_pcc,
)


class ThisTester(ModelTester):
    def _load_model(self):
        config = LlamaConfig.from_pretrained(self.model_name)
        config.num_hidden_layers = 16
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, config=config, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return model

    def _load_inputs(self):
        batch_size, seq_length = 1, 1024
        config = LlamaConfig.from_pretrained(self.model_name)
        config.num_hidden_layers = 16
        self.test_input = torch.randint(0, config.vocab_size, (batch_size, seq_length), dtype=torch.int32)
        return self.test_input


@pytest.mark.parametrize("model_name", ["meta-llama/Llama-3.1-8B"])
@pytest.mark.parametrize(
    "op_by_op",
    [None],
    ids=["full"],
)
def test_llama_8b(record_property, model_name, op_by_op):
    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    cc.enable_consteval = True

    tester = ThisTester(
        model_name,
        "eval",
        compiler_config=cc,
        assert_atol=False,
        assert_pcc=True,
        required_pcc=0.96,
        record_property_handle=record_property,
        backend="tt-experimental",
        model_name_suffix="_tt_xla",
    )
    tester.test_model()
    tester.finalize()

@pytest.mark.parametrize(
    "run_causal",
    [True, False],
)
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_llama_8b_eager(run_causal, data_type):
    torch.manual_seed(42)
    model_name = "meta-llama/Llama-3.1-8B"
    config = LlamaConfig.from_pretrained(model_name)
    config.num_hidden_layers = 7
    if run_causal:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=data_type
        ).eval()
    else:
        model = LlamaModel.from_pretrained(
            model_name, config=config, torch_dtype=data_type)
    
    batch_size, seq_length = 1, 512
    inputs = torch.randint(0, config.vocab_size, (batch_size, seq_length), dtype=torch.int32)
    outputs = model(inputs)
    if run_causal:
        cpu_outputs = outputs.logits
    else:
        cpu_outputs = outputs.last_hidden_state

    device = torch_xla.device()
    model = model.to(device)
    inputs = inputs.to(device)

    outputs = model(inputs)
    if run_causal:
        tt_outputs = outputs.logits.to("cpu")
    else:
        tt_outputs = outputs.last_hidden_state.to("cpu")

    pcc = calculate_pcc(tt_outputs, cpu_outputs)
    print(f"PCC: {pcc}")
    '''
    PCC RESULTS:
    CAUSAL MODEL:
    - Sequence Length: 1024
      - BF16: 0.7673109955645975
      - FP32:
    
    BASE MODEL:
    - Sequence Length: 1024
      - BF16: 0.07013239231878786
      - FP32:

    '''
    assert pcc >= 0.96
