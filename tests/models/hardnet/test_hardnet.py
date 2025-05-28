# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://pytorch.org/hub/pytorch_vision_hardnet/
# Reference: https://github.com/PingoLH/Pytorch-HarDNet

from PIL import Image
from torchvision import transforms
import requests
import torch
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load("PingoLH/Pytorch-HarDNet", "hardnet68", pretrained=False)
        checkpoint = "https://github.com/PingoLH/Pytorch-HarDNet/raw/refs/heads/master/hardnet68.pth"
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                checkpoint, progress=False, map_location="cpu"
            )
        )
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        url = "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png"
        input_image = Image.open(requests.get(url, stream=True).raw)
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(
            0
        )  # create a mini-batch as expected by the model
        input_batch = input_batch.to(torch.bfloat16)
        return input_batch


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "data_parallel_mode", [False, True], ids=["single_device", "data_parallel"]
)
def test_hardnet(record_property, mode, op_by_op, data_parallel_mode):
    if mode == "train":
        pytest.skip()
    model_name = "HardNet"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        if data_parallel_mode:
            pytest.skip("Op-by-op not supported in data parallel mode")
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        required_pcc=0.98,
        relative_atol=0.01,
        compiler_config=cc,
        record_property_handle=record_property,
        # TODO Enable checking - https://github.com/tenstorrent/tt-torch/issues/488
        assert_atol=False,
        data_parallel_mode=data_parallel_mode,
    )
    results = tester.test_model()

    def print_result(result):
        # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
        print(result[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(results[0], dim=0)
        print(probabilities)

    if mode == "eval":
        ModelTester.print_outputs(results, data_parallel_mode, print_result)

    tester.finalize()
