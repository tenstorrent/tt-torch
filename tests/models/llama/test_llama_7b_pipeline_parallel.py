# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn as nn
import tt_torch
from tt_torch.tools.device_manager import DeviceManager
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import BackendOptions
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from accelerate import infer_auto_device_map
from tt_torch.tools.verify import verify_against_golden
from tests.utils import ModelTester


def get_model_and_tokenizer(model_name):
    # Download model from cloud
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", torch_dtype=torch.bfloat16
    )
    tokenizer.pad_token = tokenizer.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    for param in m.parameters():
        param.requires_grad = False
    return m, tokenizer


def get_sample_input(tokenizer, test_input):
    inputs = tokenizer.encode_plus(
        test_input,
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )
    return inputs


def decode_output(outputs, tokenizer):
    next_token_logits = outputs.logits[:, -1]
    next_token = next_token_logits.softmax(dim=-1).argmax()
    return tokenizer.decode([next_token])


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize("model_name", ["huggyllama/llama-7b"])
def test_llama_7b_pipeline_parallel(record_property, model_name, mode):
    prompt = "I enjoy walking in the"
    model, tokenizer = get_model_and_tokenizer(model_name)
    test_input = get_sample_input(tokenizer, prompt)
    parent_device = DeviceManager.create_parent_mesh_device([1, 2])

    # Create submeshes that target different devices
    device1 = DeviceManager.create_sub_mesh_device(parent_device, (0, 0))
    device2 = DeviceManager.create_sub_mesh_device(parent_device, (0, 1))

    dont_split = (
        model._no_split_modules if hasattr(model, "_no_split_modules") else None
    )
    # The devices have 12GB of memory each, but we set it to 11GB to ensure
    # there's enough room for activation tensors and other overhead.
    device_map = infer_auto_device_map(
        model, max_memory={0: "11GiB", 1: "11GiB"}, no_split_module_classes=dont_split
    )

    required_atol = 0.1
    assert_pcc = True
    assert_atol = False

    options = BackendOptions()
    cc = CompilerConfig()
    options.compiler_config = cc
    cc.device_map = device_map
    cc.enable_consteval = True
    options.devices = [device1, device2]
    compiled_model = torch.compile(
        model, backend="tt-legacy", dynamic=False, options=options
    )
    out = compiled_model(**test_input)
    golden = model(**test_input)
    pccs, atols, _, _, _ = verify_against_golden(
        tuple([golden.logits]),
        tuple([out.logits]),
        assert_pcc,
        assert_atol,
        required_atol=required_atol,
    )

    ModelTester.GenerateCustomTestReport(
        record_property,
        model_name,
        cc,
        pccs,
        atols,
        required_atol=required_atol,
        assert_pcc=assert_pcc,
        assert_atol=assert_atol,
    )

    DeviceManager.release_sub_mesh_device(device1)
    DeviceManager.release_sub_mesh_device(device2)
    DeviceManager.release_parent_device(parent_device)
