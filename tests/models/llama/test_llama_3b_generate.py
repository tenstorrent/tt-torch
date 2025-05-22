# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, StaticCache


class ThisTester(ModelTester):
    def _load_model(self):
        
        # note - use_cache in llamaconfig is set to default to true
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, use_cache=True
        )
        model.config.num_hidden_layers = 2 # otherwise it's too big to fit on device
                
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
                
        return model

    def _load_inputs(self):
        self.test_input = "This is a sample text from "
        
        input_length = 15
        inputs = self.tokenizer.encode_plus(
            self.test_input,
            return_tensors="pt",
            max_length=input_length,
            padding="max_length",
            truncation=True,
        )
        
        # setup static cache
        batch_size = 1
        max_cache_len = 64
        static_cache = StaticCache(
            config=self.framework_model.config,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=self.framework_model.device,
            dtype=self.framework_model.dtype,
        )
        
        cache_position = torch.arange(0, input_length)
        args = {"input_ids": inputs.input_ids, "past_key_values": static_cache, "use_cache": True, "cache_position": cache_position}    
        
        return args

    def set_model_eval(self, model):
        return model


def test_llama_3b(record_property):
    cc = CompilerConfig()
    cc.compile_depth = CompileDepth.EXECUTE
    cc.enable_consteval = True
    cc.consteval_parameters = True
    mode="eval"
    model_name = 'meta-llama/Llama-3.2-3B'
    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        assert_atol=False,
        assert_pcc=False,
        record_property_handle=record_property,
    )
    
    results = tester.test_model()
    tester.finalize()
