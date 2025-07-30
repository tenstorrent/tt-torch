# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
import torch_xla.core.xla_model as xm

from tt_torch.tools.utils import calculate_pcc
from third_party.tt_forge_models.mistral.pytorch import ModelLoader, ModelVariant


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs()


@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.TORCH, None],
    ids=["op_by_op_torch", "full"],
)
def test_mistral(record_property, op_by_op):
    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    cc.enable_consteval = True

    loader = ModelLoader(variant=ModelVariant.MINISTRAL_3B)
    model_info = loader.get_model_info(variant=ModelVariant.MINISTRAL_3B)
    model_name = model_info.name
    model_group = model_info.group.value

    # TODO Enable PCC/ATOL/Checking - https://github.com/tenstorrent/tt-torch/issues/689
    tester = ThisTester(
        model_name,
        "eval",
        loader=loader,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_atol=False,
        assert_pcc=False,
        model_group=model_group,
        backend="tt-experimental",
        model_name_suffix="_tt_xla",
    )
    tester.test_model()
    tester.finalize()


def test_mistral3b_eager():
    loader = ModelLoader(variant=ModelVariant.MINISTRAL_3B)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs()

    cpu_outputs = model(**inputs).logits

    device = xm.xla_device()

    model = model.to(device)
    inputs = inputs.to(device)

    tt_outputs = model(**inputs).logits.to("cpu")

    pcc = calculate_pcc(tt_outputs, cpu_outputs)
    print(f"PCC: {pcc}")
