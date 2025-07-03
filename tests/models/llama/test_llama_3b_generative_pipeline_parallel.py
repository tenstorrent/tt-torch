# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig
from tt_torch.tools.device_manager import DeviceManager
from tt_torch.dynamo.backend import backend, BackendOptions
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StaticCache,
)
from tests.utils import clear_dynamo_cache
from accelerate import infer_auto_device_map


class PrefillTester(ModelTester):
    def _load_model(self):
        # set up the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            use_cache=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.model.eval()

    def _load_inputs(self):
        self.test_input = "This is a sample text from "
        self.max_cache_len = 64 + 64
        batch_size = 1
        inputs = self.tokenizer.encode_plus(
            self.test_input,
            return_tensors="pt",
            truncation=True,
        )

        # set up static cache
        static_cache = StaticCache(
            config=self.model.config,
            max_batch_size=batch_size,
            max_cache_len=self.max_cache_len,
            device=self.model.device,
            dtype=self.model.dtype,
        )

        cache_position = torch.arange(0, inputs.input_ids.shape[1])

        args = {
            "input_ids": inputs.input_ids,
            "past_key_values": static_cache,
            "use_cache": True,
            "cache_position": cache_position,
        }
        return args


@torch.inference_mode()
def test_llama_3b_generative_pipeline_parallel(record_property):
    clear_dynamo_cache()
    cc = CompilerConfig()
    cc.enable_consteval = False
    cc.consteval_parameters = False
    cc.dump_debug = True
    cc.dump_info = True

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

    model = tester._load_model()
    tokenizer = tester.tokenizer
    input_args = tester._load_inputs()
    generated_ids = input_args["input_ids"]
    print(tokenizer.decode(generated_ids[0].tolist()), end="", flush=True)

    parent_device = DeviceManager.create_parent_mesh_device([1, 2])

    # Create submeshes that target different devices
    device1 = DeviceManager.create_sub_mesh_device(parent_device, (0, 0))
    device2 = DeviceManager.create_sub_mesh_device(parent_device, (0, 1))

    dont_split = (
        model._no_split_modules if hasattr(model, "_no_split_modules") else None
    )
    device_map = infer_auto_device_map(
        model, max_memory={0: "5GiB", 1: "5GiB"}, no_split_module_classes=dont_split
    )

    options = BackendOptions()
    options.compiler_config = cc
    cc.device_map = device_map
    options.devices = [device1, device2]

    buffer_cache = {}
    options.buffer_cache = buffer_cache

    constant_cache = {}
    options.constant_cache = constant_cache

    compiled_model = torch.compile(
        model, backend=backend, dynamic=False, options=options
    )

    # up to _global_max_cache_len - input_args["input_ids"].shape[1]
    tokens_to_generate = 32

    for i in range(tokens_to_generate):
        outputs = compiled_model(**input_args)
        next_token_ids = outputs.logits[:, -1:].argmax(dim=-1)
        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
        print(tokenizer.decode(next_token_ids[0].tolist()), end="", flush=True)

        cache_position = input_args["cache_position"][-1:] + 1
        input_args = {
            "input_ids": next_token_ids.to(dtype=torch.int32),
            "past_key_values": input_args["past_key_values"],  # updated in place
            "cache_position": cache_position,
            "use_cache": True,
        }
    print()  # Add a newline at the end of the output
    DeviceManager.release_sub_mesh_device(device1)
    DeviceManager.release_sub_mesh_device(device2)
    DeviceManager.release_parent_device(parent_device)
    clear_dynamo_cache()
