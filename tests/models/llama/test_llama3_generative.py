# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from tt_torch.tools.verify import calculate_pcc
from tt_torch.tools.device_manager import DeviceManager
from tt_torch.dynamo.backend import backend, BackendOptions
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StaticCache,
)
from tests.utils import clear_dynamo_cache
import tt_mlir

_global_max_cache_len = 64 + 64


class ThisTester(ModelTester):
    def _load_model(self):
        model_name = "meta-llama/Llama-3.2-3B"
        # set up the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            use_cache=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.model

    def _load_inputs(self):
        inputs = self.tokenizer.encode_plus(
            "I like taking walks in the",
            return_tensors="pt",
            truncation=True,
            return_attention_mask=True,
        )

        # set up static cache
        static_cache = StaticCache(
            config=self.model.config,
            max_batch_size=1,
            max_cache_len=_global_max_cache_len,
            device=self.model.device,
            dtype=self.model.dtype,
        )

        cache_position = torch.arange(0, inputs.input_ids.shape[1])

        args = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": 32,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "past_key_values": static_cache,
            "use_cache": True,
            "cache_position": cache_position,
        }
        return args


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_llama3_generate(record_property, mode, op_by_op):
    model_name = "meta-llama/Llama-3.2-3B"

    # Setup compilation
    cc = CompilerConfig()

    # Consteval disabled due to 4D Causal Attention Mask evaluation getting constant folded in torchfx
    #   due to incorrect tracing of static cache and malformed output missing static cache tensors
    cc.enable_consteval = False
    cc.consteval_parameters = False

    options = BackendOptions()
    options.compiler_config = cc

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=True,
        assert_atol=False,
        run_generate=True,
        backend="tt",
    )

    results = tester.test_model()

    decoded_output = tester.tokenizer.decode(results[0], skip_special_tokens=True)
    print(decoded_output)

    # device = DeviceManager.create_parent_mesh_device(mesh_shape=[1, 1])
    # options.devices = [device]

    # compiled_model = torch.compile(
    #     model, backend=backend, dynamic=False, options=options
    # )

    # generation_start = time.time()
    # generated_ids = compiled_model.generate(**input_args)
    # total_generation_time = time.time() - generation_start

    # generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # print(f"Generated text: '{generated_text}'")
    # print(f"Total generation time: {total_generation_time:.3f}s")
    # print(f"Tokens generated: {generated_ids.shape[1] - input_args['input_ids'].shape[1]}")

    # # Cleanup
    # DeviceManager.release_parent_device(device)
    # clear_dynamo_cache()
