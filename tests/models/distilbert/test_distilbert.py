# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from tt_torch.tools.device_manager import DeviceManager


class ThisTester(ModelTester):
    def _load_model(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        model = DistilBertModel.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        return model

    def _load_inputs(self):
        self.text = "Transformers provide state-of-the-art results in NLP."
        inputs = self.tokenizer(self.text, return_tensors="pt")
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize("model_name", ["distilbert-base-uncased"])
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_distilbert(record_property, model_name, mode, op_by_op):

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
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()

    if mode == "eval":
        print(f"Model: {model_name} | Input: {tester.text} | Output: {results}")

    tester.finalize()


@pytest.mark.parametrize(
    "num_loops",
    [64],
)
@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize("model_name", ["distilbert-base-uncased"])
@pytest.mark.parametrize(
    "op_by_op",
    [None],
    ids=["full"],
)
def test_distilbert_multiloop(record_property, model_name, mode, op_by_op, num_loops):
    import time

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.cache_preprocessed_constants = True
    devices = DeviceManager.get_available_devices(
        mesh_shape=[1, 1], enable_async_ttnn=True
    )
    assert len(devices) == 1, (
        "Failed to get available devices"
        if len(devices) == 0
        else "More than 1 device taken"
    )
    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
        model_name_suffix="-multiloop",
        device=devices[0],
    )
    model = tester.compile_model(tester.get_framework_model(), tester.compiler_config)

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_loops):
            results = tester.run_model(model, tester.inputs)
        end_time = time.time()

        print(f"Model: {model_name} | Input: {tester.text} | Output: {results}")
        print(f"{num_loops} iterations took {(end_time - start_time)} seconds")

    tester.finalize()
    DeviceManager.release_devices()
