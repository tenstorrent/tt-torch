# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/linear_autoencoder/pytorch_linear_autoencoder.py

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class LinearAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder_lin1 = torch.nn.Linear(784, 128)
        self.encoder_lin2 = torch.nn.Linear(128, 64)
        self.encoder_lin3 = torch.nn.Linear(64, 12)
        self.encoder_lin4 = torch.nn.Linear(12, 3)

        # Decoder
        self.decoder_lin1 = torch.nn.Linear(3, 12)
        self.decoder_lin2 = torch.nn.Linear(12, 64)
        self.decoder_lin3 = torch.nn.Linear(64, 128)
        self.decoder_lin4 = torch.nn.Linear(128, 784)

        # Activation Function
        self.act_fun = torch.nn.ReLU()

    def forward(self, x):
        # Encode
        act = self.encoder_lin1(x)
        act = self.act_fun(act)
        act = self.encoder_lin2(act)
        act = self.act_fun(act)
        act = self.encoder_lin3(act)
        act = self.act_fun(act)
        act = self.encoder_lin4(act)

        # Decode
        act = self.decoder_lin1(act)
        act = self.act_fun(act)
        act = self.decoder_lin2(act)
        act = self.act_fun(act)
        act = self.decoder_lin3(act)
        act = self.act_fun(act)
        act = self.decoder_lin4(act)

        return act


class ThisTester(ModelTester):
    def _load_model(self):
        # Instantiate model
        # NOTE: The model has not been pre-trained or fine-tuned.
        # This is for demonstration purposes only.
        model = LinearAE()
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        # Define transform: convert to tensor, normalize, flatten to (1, 1, 784)
        transform = transforms.Compose([
            transforms.ToTensor(),  # (1, 28, 28)
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(1, 1, -1))  # Flatten to (1, 1, 784)
        ])

        # Load dataset and take first 32 samples
        dataset = load_dataset("mnist")["train"]
        samples = [dataset[i]["image"] for i in range(32)]
        sample_tensors = [transform(img) for img in samples]  # Each is (1, 1, 784)

        # Stack into 4D tensor: (32, 1, 1, 784)
        batch_tensor = torch.stack(sample_tensors, dim=0)
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
def test_autoencoder_linear(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "Autoencoder (linear)"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.dump_info = True
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

    tester.finalize()
