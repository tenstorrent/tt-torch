# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://huggingface.co/facebook/seamless-m4t-v2-large

import torch
import pytest

from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.seamless_m4t.pytorch import ModelLoader


class ThisTester(ModelTester):
    def _load_model(self):
        return self.loader.load_model()

    def _load_inputs(self):
        return self.loader.load_inputs()


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_seamless_m4t(record_property, mode, op_by_op):
    loader = ModelLoader(variant=None)
    model_info = loader.get_model_info(variant=None)
    model_name = model_info.name
    model_group = model_info.group.value

    cc = CompilerConfig()
    cc.enable_consteval = True
    # cc.consteval_parameters = True
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
        assert_atol=False,
        run_generate=False,
        model_group=model_group,
    )

    results = tester.test_model()

    if mode == "eval":
        # Use loader's decode_output method
        decoded_output = loader.decode_output(results)

        print(
            f"""
        model_name: {model_name}
        {decoded_output}
        """
        )

        if tester.run_generate:
            sample_rate = tester.model.config.sampling_rate
            # uncomment this to download the output audio
            # scipy.io.wavfile.write(
            #     "out_from_text.wav", rate=sample_rate, data=results[0].numpy().squeeze()
            # )

    tester.finalize()
