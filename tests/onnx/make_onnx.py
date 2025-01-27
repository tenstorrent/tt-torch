# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, 2)
        output = F.softmax(x, dim=1)
        return output


def main():
    model = Net()
    input_names = ["image"]
    output_names = ["prediction"]
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        "/localdev/achoudhury/tt-torch/tests/onnx/mnist_custom.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
    )
    print("Model exported to mnist_custom.onnx")


main()
