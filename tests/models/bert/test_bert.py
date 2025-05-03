# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.bert import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return ModelLoader.load_model()

    def _load_inputs(self):
        return ModelLoader.load_inputs()


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_bert(record_property, mode, op_by_op):
    model_name = "BERT"

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
        relative_atol=0.012,
        compiler_config=cc,
        record_property_handle=record_property,
        # TODO Enable checking - https://github.com/tenstorrent/tt-torch/issues/489
        assert_pcc=False,
        assert_atol=False,
    )
    results = tester.test_model()

    if mode == "eval":
        answer = ModelLoader.decode_output(results, tester.inputs)

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
