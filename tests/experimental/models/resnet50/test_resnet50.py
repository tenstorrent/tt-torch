# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch_xla
import torchvision.models as models
from PIL import Image

import pytest
from tests.utils import ModelTester
from tt_torch.dynamo.backend import BackendOptions
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file

from tt_torch.tools.utils import (
    calculate_pcc,
)

import time


class ThisTester(ModelTester):
    def _load_model(self):
        # Load the ResNet-50 model with updated API
        self.weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=self.weights)
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        # Define a transformation to preprocess the input image using the weights transforms
        preprocess = self.weights.transforms()

        # Load and preprocess the image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0)
        batch_t = batch_t.to(torch.bfloat16)
        return batch_t


@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.TORCH, None],
    ids=["op_by_op_torch", "full"],
)
def test_resnet(record_property, op_by_op):
    model_name = "ResNet50"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP

    assert_pcc = True
    required_pcc = 0.97

    tester = ThisTester(
        model_name,
        "eval",
        required_atol=0.03,
        required_pcc=required_pcc,
        compiler_config=cc,
        assert_pcc=assert_pcc,
        assert_atol=False,
        record_property_handle=record_property,
        data_parallel_mode=False,
        backend="tt-experimental",
        model_name_suffix="_tt_xla",
    )

    results = tester.test_model()

    def print_result(result):
        _, indices = torch.topk(result, 5)
        print(f"Top 5 predictions: {indices[0].tolist()}")

    ModelTester.print_outputs(results, False, print_result)

    tester.finalize()


def test_resnet_eager():
    # Eager mode does not require toggling the experimental path
    # as this will use no torch.compile infrastructure and directly
    # interacts with PJRT

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = model.to(torch.bfloat16).eval()

    preprocess = weights.transforms()

    # Load and preprocess the image
    image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(str(image_file))
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0)
    batch_t = batch_t.to(torch.bfloat16)

    cpu_result = model(batch_t)

    # Push model and input to device
    model = model.to("xla")
    batch_t = batch_t.to("xla")

    tt_result = model(batch_t).to("cpu")

    _, tt_indices = torch.topk(tt_result, 5)
    _, cpu_indices = torch.topk(cpu_result, 5)
    print(f"Top 5 predictions on TT device: {tt_indices[0].tolist()}")
    print(f"Top 5 predictions on CPU device: {cpu_indices[0].tolist()}")

    pcc = calculate_pcc(tt_result, cpu_result)
    assert pcc >= 0.98, f"Failed with pcc {pcc}"


def test_resnet_perf():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = model.to(torch.bfloat16).eval()

    preprocess = weights.transforms()

    # Load and preprocess the image
    image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(str(image_file))
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0)
    batch_t = batch_t.to(torch.bfloat16)

    cpu_result = model(batch_t)

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.push_outputs_to_cpu = False
    options = BackendOptions(compiler_config=cc)
    # Push model and input to device
    with torch.inference_mode():
        model = torch.compile(model, backend="tt-experimental", options=options)
        print("Beginning compile kernels (1st iteration)")
        tt_result = model(batch_t)
        tt_result = tt_result.to("cpu")  # Blocks

        print("Finished compile kernels")

        num_inference_calls = 10
        results = [None] * num_inference_calls
        print(f"Beginning {num_inference_calls} inference calls")
        start = time.time()
        for i in range(num_inference_calls):
            results[i] = model(batch_t)

        for i in range(num_inference_calls):
            results[i] = results[i].to("cpu")
        end = time.time()

        print(
            f"Time taken: {end - start} seconds. FPS: {num_inference_calls / (end - start)}"
        )

    # for tt_result in results:
    _, tt_indices = torch.topk(results[-1], 5)
    print(f"Top 5 predictions on TT device: {tt_indices[0].tolist()}")

    _, cpu_indices = torch.topk(cpu_result, 5)
    print(f"Top 5 predictions on CPU device: {cpu_indices[0].tolist()}")

    # for i in range(num_inference_calls):
    #     print(f"Time waiting for {i} = {time_waiting[i]}")

    pcc = calculate_pcc(tt_result, cpu_result)
    assert pcc >= 0.98, f"Failed with pcc {pcc}"


# Empty property record_property
def empty_record_property(a, b):
    pass


# Run pytorch implementation
if __name__ == "__main__":
    test_resnet(empty_record_property)
