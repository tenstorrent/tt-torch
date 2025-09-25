# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester
from tt_torch.tools.device_manager import DeviceManager
from tt_torch.tools.utils import CompilerConfig, OpByOpBackend
from tt_torch.dynamo.backend import BackendOptions
from accelerate import infer_auto_device_map
from tt_torch.tools.verify import verify_against_golden


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", torch_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def load_inputs(tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
    return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "tiiuae/Falcon3-7B-Base",
        "tiiuae/Falcon3-10B-Base",
    ],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_falcon_pipeline_parallel(record_property, model_name, mode, op_by_op):

    prompt = "Hey, are you conscious? Can you talk to me?"
    model, tokenizer = load_model(model_name)
    test_input = load_inputs(tokenizer, prompt)
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
    required_pcc = 0.98 if model_name == "tiiuae/Falcon3-10B-Base" else 0.99
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
        assert_pcc=assert_pcc,
        assert_atol=assert_atol,
        required_atol=required_atol,
        required_pcc=required_pcc,
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
        model_group="red",
    )

    DeviceManager.release_sub_mesh_device(device1)
    DeviceManager.release_sub_mesh_device(device2)
    DeviceManager.release_parent_device(parent_device)
