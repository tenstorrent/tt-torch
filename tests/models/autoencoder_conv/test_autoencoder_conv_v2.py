# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/conv_autoencoder/pytorch_conv_autoencoder.py

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ConvAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder_conv2d_1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.encoder_conv2d_2 = torch.nn.Conv2d(16, 4, 3, padding=1)
        self.encoder_max_pool2d = torch.nn.MaxPool2d(2, 2)

        # Decoder
        self.decoder_conv2d_1 = torch.nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.decoder_conv2d_2 = torch.nn.ConvTranspose2d(16, 1, 2, stride=2)

        # Activation Function
        self.act_fun = torch.nn.ReLU()

    def forward(self, x):
        # Encode
        act = self.encoder_conv2d_1(x)
        act = self.act_fun(act)
        act = self.encoder_max_pool2d(act)
        act = self.encoder_conv2d_2(act)
        act = self.act_fun(act)
        act = self.encoder_max_pool2d(act)

        # Decode
        act = self.decoder_conv2d_1(act)
        act = self.act_fun(act)
        act = self.decoder_conv2d_2(act)

        return act


class ThisTester(ModelTester):
    def _load_model(self):
        # Instantiate model
        # NOTE: The model has not been pre-trained or fine-tuned.
        # This is for demonstration purposes only.
        model = ConvAE()
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        # Define transform to normalize data
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Load sample from MNIST dataset
        dataset = load_dataset("mnist")
        sample = dataset["train"][0]["image"]
        n_sample_tensor = [transform(sample).unsqueeze(0)]
        batch_tensor = torch.cat(n_sample_tensor, dim=0)
        batch_tensor = batch_tensor.to(torch.bfloat16)
        return batch_tensor


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_autoencoder_conv_v2(record_property, mode, op_by_op):
    model_name = f"Autoencoder (conv)"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name, mode, compiler_config=cc, record_property_handle=record_property
    )
    results = tester.test_model()

    if mode == "eval":
        print("Output: ", results)
