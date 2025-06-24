# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.bert.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model(dtype_override=torch.bfloat16)

    def _load_inputs(self):
        return self.loader.load_inputs()


# Print available variants for reference
available_variants = ModelLoader.query_available_variants()
print("Available variants:", available_variants)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "variant_info",
    available_variants.items(),
    ids=list(available_variants.keys()),
)
def test_bert(record_property, mode, op_by_op, variant_info):

    # Use variant in model name if specified
    variant, variant_config = variant_info
    loader = ModelLoader(variant=variant)
    model_info = loader.get_model_info()
    model_name = model_info.name

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        loader=loader,
        relative_atol=0.012,
        compiler_config=cc,
        record_property_handle=record_property,
        assert_pcc=True,
        assert_atol=False,
    )
    results = tester.test_model()

    if mode == "eval":
        answer = loader.decode_output(results, tester.inputs)

        print(
            f"""
        model_name: {model_name}
        input:
            context: {ModelLoader.context}
            question: {ModelLoader.question}
        answer: {answer}
        """
        )

    tester.finalize()
