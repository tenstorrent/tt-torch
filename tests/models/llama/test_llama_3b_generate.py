# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    StaticCache,
)
import tt_mlir

from tt_torch.tools.verify import calculate_pcc
class PrefillTester(ModelTester):
    def _load_model(self):

        # note - use_cache in llamaconfig is set to default to true
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            use_cache=True,
        )
        model.config.num_hidden_layers = 2

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return model

    def _load_inputs(self):
        self.test_input = "This is a sample text from "

        input_length = 32
        inputs = self.tokenizer.encode_plus(
            self.test_input,
            return_tensors="pt",
            # max_length=input_length,
            # padding="max_length",
            truncation=True,
        )

        # max pad + cache_position side effects cause a different graph to be synthesized...

        # setup static cache
        batch_size = 1
        max_cache_len = 64
        static_cache = StaticCache(
            config=self.framework_model.config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=self.framework_model.device,
            dtype=self.framework_model.dtype,
        )

        # attention_mask = inputs.attention_maskmask
        # cache_position = attention_mask.cumsum(dim=-1) - 1
        # cache_position = cache_position.masked_fill(attention_mask == 0, 0)

        cache_position = torch.arange(0, inputs.input_ids.shape[1])
        print(f"[James] @load_inputs dtype of input_ids - {inputs.input_ids.dtype}")
        args = {
            "input_ids": inputs.input_ids.to(dtype=torch.int32),  # Convert to int32 for compatibility
            "past_key_values": static_cache,
            "use_cache": True,
            "cache_position": cache_position,
            # "attention_mask": attention_mask,
            # "position_ids": cache_position.unsqueeze(0),  # Assuming batch size of 1
        }
        return args

    @torch.inference_mode()
    def get_torchcompiled_gm(self, runtime_tensor_cache):
        model = self.get_framework_model()
        # golden = self.get_golden_outputs(model, self.inputs)

        model = self.compile_model(model, self.compiler_config, data_parallel_mode=False, runtime_tensor_cache=runtime_tensor_cache)

        return model

        # outputs = self.run_model(model, self.inputs)
        # self.record_property("achieved_compile_depth", "EXECUTE")

        # if self.compiler_config._enable_intermediate_verification:
        #     self.verify_intermediates_after_execution()

        # self._verify_full_execution_output(outputs, golden, assert_eval_token_mismatch)
        # return outputs

    @torch.inference_mode()
    def run_model_with_inputs(self, model, inputs):
        # Run the model with the provided inputs
        outputs = model(**inputs)
        return outputs

@torch.inference_mode()
def test_llama_3b(record_property):
    cc = CompilerConfig()
    cc.compile_depth = CompileDepth.EXECUTE
    cc.enable_consteval = False
    cc.consteval_parameters = False
    mode = "eval"
    model_name = "meta-llama/Llama-3.2-3B"
    
    tester = PrefillTester(
        model_name,
        mode,
        compiler_config=cc,
        assert_atol=False,
        assert_pcc=False,
        record_property_handle=record_property,
    )

    input_args = tester._load_inputs()

    input_ids = input_args["input_ids"]
    static_cache = input_args["past_key_values"]
    cache_position = input_args["cache_position"]
    # attention_mask = input_args["attention_mask"]

    # compile prefill fx graph to flatbuffer and run

    max_new_tokens = 10
    
    runtime_tensor_cache = {}
    print("Runtime tensor cache id: ", id(runtime_tensor_cache))
    gm = tester.get_torchcompiled_gm(runtime_tensor_cache)
    
    compare_golden = False

    generated_ids = input_ids
    for i in range(max_new_tokens):
        print("\n===== Decode step", i, "=====\n")
        print(f"Input args to step {i}", input_args)
        
        
        outputs = tester.run_model_with_inputs(gm, input_args)    
        next_token_ids = outputs.logits[:, -1:].argmax(dim=-1)
        
        # print(f"Cache after step {i}: {list(runtime_tensor_cache.keys())}")

        if compare_golden:
            golden = tester.get_golden_outputs(tester.get_framework_model(), input_args)        
            golden_next_token_ids = golden.logits[:, -1:].argmax(dim=-1)
            golden_static_cache_tensors = golden.past_key_values.key_cache
            golden_static_cache_tensors.extend(golden.past_key_values.value_cache)
            # print(f"Next token id prediction for decode step {i}: {next_token_ids.item()}. Golden: {golden_next_token_ids.item()}.")
            print(f"Next token id prediction for decode step {i}: {next_token_ids.item()}. Golden: {golden_next_token_ids.item()}. PCC : {calculate_pcc(torch.flatten(outputs.logits), torch.flatten(golden.logits))}")
        
            for key, value in runtime_tensor_cache.items():
                host_cache = tt_mlir.to_host(value)[0] # returns single element tuple, I think
                # print(host_cache)
                print(f"Runtime tensor cache key: {key}")
                print(f"{torch.mean(host_cache[0,0,:,:], dim=-1).tolist()}")
                
                # Rake & Sweep for golden verification of static caches
                rake_pcc = [] 
                for golden_tensor in golden_static_cache_tensors:
                    rake_pcc.append(calculate_pcc(host_cache,golden_tensor))
                
                if rake_pcc:
                    best_idx = rake_pcc.index(max(rake_pcc))
                    best_value = rake_pcc[best_idx]
                    
                    # Format with the best value highlighted
                    pcc_formatted = [f"{pcc:.4f}" for pcc in rake_pcc]
                    pcc_formatted[best_idx] = f"\033[92m{best_value:.4f}\033[0m"  # Highlight best in green
                    
                    pcc_str = "[" + ", ".join(pcc_formatted) + "]"
                    print(f"PCC matches: (best: {best_value:.4f} at idx {best_idx})")
                        
                    

        #  = outputs.past_key_values
        # -> are past_key_values updated in place?
        # eg. do I need to update the static_cache object with new past_key_values?

        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
        print(
            "\033[91mDecoded output so far: ",
            tester.tokenizer.decode(generated_ids[0].tolist()),
            "\033[0m",
        )

        # attention_mask = input_args["attention_mask"]
        # attention_mask = torch.cat(
        #     [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
        #     dim=-1,
        # )
        cache_position = cache_position[-1:] + 1

        # inspect static_cache
        # key_cache0_head0_vals = torch.mean(static_cache.key_cache_0[0, 0, :, :], dim=-1)
        # print("Key cache slice along seqlen: ", key_cache0_head0_vals.tolist())

        # print("next token_ids dtype before autocast: ", next_token_ids.dtype)
        # next_token_ids = next_token_ids.clone().to(dtype=torch.int32)
        # print("next token_ids dtype after autocast: ", next_token_ids.dtype)
        
        val = next_token_ids.item()
        clean_ids = torch.tensor([[val]], dtype=torch.int32)
        embedding_layer = tester.get_framework_model().get_input_embeddings()
        next_token_embeds = embedding_layer(clean_ids)
        
        input_args = {
            "input_ids": next_token_ids.to(dtype=torch.int32), 
            # "input_ids": torch.tensor([[220]], dtype=torch.int32).detach().clone(),  # For testing, use a fixed input
            "past_key_values": static_cache,  # assume in place update for now
            "use_cache": True,
            "cache_position": cache_position,
            # "position_ids": cache_position.unsqueeze(0),  # Assuming batch size of 1
            # "attention_mask": attention_mask,
        }

        # use embedddings instead.
        # print("[James] Force use imputs embeds")
        # del input_args['input_ids']
        # input_args['inputs_embeds'] = next_token_embeds
        
        # if i == 0:
        #     torch._dynamo.reset()

