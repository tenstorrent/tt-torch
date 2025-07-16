# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/Qwen/Qwen2.5-1.5B
import torch
import pytest

# Load model directly
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
import tt_mlir
from third_party.tt_forge_models.qwen.casual_lm.pytorch import ModelLoader
import torch_xla.core.xla_model as xm

from tt_torch.tools.utils import (
    calculate_pcc,
)


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs(dtype_override=torch.bfloat16)


@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.TORCH, None],
    ids=["op_by_op_torch", "full"],
)
def test_qwen2_causal_lm(record_property, op_by_op):
    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    cc.enable_consteval = True

    # TODO: Remove this once PCC ATOL is fixed on blackhole runners - https://github.com/tenstorrent/tt-torch/issues/1003
    assert_pcc = tt_mlir.get_arch() != tt_mlir.Arch.BLACKHOLE

    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)

    tester = ThisTester(
        model_info.name,
        "eval",
        loader=loader,
        model_info=model_info,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=assert_pcc,
        assert_atol=False,
        run_generate=False,
        required_pcc=0.85,
        backend="tt-experimental",
    )

    results = tester.test_model()

    gen_text = loader.decode_output(results, dtype_override=torch.bfloat16)

    print(f"Model: {model_info.name} | Input: {loader.text} | Decoded Text: {gen_text}")

    tester.finalize()


def test_qwen2_causal_lm_eager():
    loader = ModelLoader(variant=None)

    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    cpu_outputs = model(**inputs).logits

    device = xm.xla_device()
    inputs["input_ids"] = inputs["input_ids"].to(device)
    model = model.to(device)

    tt_outputs = model(**inputs).logits.to("cpu")

    gen_text_cpu = loader.decode_output((cpu_outputs,))
    gen_text_tt = loader.decode_output((tt_outputs,))

    print(f'CPU Decoded Text: "{gen_text_cpu}"')
    print(f'TT Decoded Text: "{gen_text_tt}"')

    pcc = calculate_pcc(tt_outputs, cpu_outputs)
    print(f"PCC: {pcc}")

    assert pcc >= 0.85
