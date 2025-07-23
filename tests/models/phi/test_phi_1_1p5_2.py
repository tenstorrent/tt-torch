# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Phi 1: https://huggingface.co/microsoft/phi-1
# Phi 1.5: https://huggingface.co/microsoft/phi-1_5
# Phi 2: https://huggingface.co/microsoft/phi-2

import torch
import pytest

from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.phi.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs()


# Print available variants for reference
available_variants = ModelLoader.query_available_variants()
print("Available variants: ", [str(k) for k in available_variants.keys()])


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "variant,variant_config",
    available_variants.items(),
    ids=[str(k) for k in available_variants.keys()],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_phi(record_property, variant, variant_config, mode, op_by_op):
    loader = ModelLoader(variant=variant)
    model_info = loader.get_model_info(variant=variant)
    model_name = model_info.name
    model_group = model_info.group.value

    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        loader=loader,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group=model_group,
        required_pcc=0.85
        if variant.value == "1"
        else 0.92,  # PCC drop observed around Jul 17, follow up in https://github.com/tenstorrent/tt-torch/issues/1070
        run_generate=False,
        assert_atol=False,
    )

    results = tester.test_model()

    if mode == "eval":
        # Use loader's decode_output method
        decoded_output = loader.decode_output(results, dtype_override=torch.bfloat16)

        # Get test input from loader for display
        test_input = '''def print_prime(n):
                        """
                        Print all primes between 1 and n
                        """'''

        print(
            f"""
        model_name: {model_name}
        input: {test_input}
        output: {decoded_output}
        """
        )
    tester.finalize()
