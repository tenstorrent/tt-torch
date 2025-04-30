# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import backend
from PIL import Image
from torchvision import transforms, datasets
import torchvision.models as models
import tabulate
import requests
import time
import tt_mlir
from tt_torch.tools.device_manager import DeviceManager
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py
class MnistModel(torch.nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


model = MnistModel().to(torch.float16).eval()


def main():
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    num_devices = DeviceManager.get_num_available_devices()
    parent, devices = DeviceManager.acquire_available_devices(enable_async_ttnn=True)

    tt_models = []
    for device in devices:
        options = {}
        options["compiler_config"] = cc
        options["device"] = device
        options["async_mode"] = True
        tt_models.append(
            torch.compile(model, backend=backend, dynamic=False, options=options)
        )

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    dataloader = DataLoader(test_dataset, batch_size=1)
    test_inputs = []
    expected_outputs = []
    for i in range(num_devices):
        test_input, expected_output = next(iter(dataloader))
        test_input = test_input.to(torch.float16)
        test_inputs.append(test_input)
        expected_outputs.append(expected_output)

    runtime_tensors_list = []
    for i in range(num_devices):
        tt_model = tt_models[i]
        test_input = test_inputs[i]
        start = time.time()
        runtime_tensors = tt_model(test_input)
        end = time.time()
        print("DEBUG - TIME FOR RUNTIME TENSOR: ", end - start)
        runtime_tensors_list.append(runtime_tensors)

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i in range(len(runtime_tensors_list)):
            runtime_tensors = runtime_tensors_list[i]
            expected_output = expected_outputs[i]
            start = time.time()
            curr_output = tt_mlir.to_host(runtime_tensors)
            end = time.time()
            print("DEBUG - TIME FOR TO_HOST: ", end - start)
            test_loss += F.nll_loss(
                curr_output, expected_output, size_average=False
            ).item()
            pred = curr_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(expected_output.view_as(pred)).sum().item()

    test_loss /= len(test_inputs)

    DeviceManager.release_parent_device(parent, cleanup_sub_devices=True)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_inputs), 100.0 * correct / len(test_inputs)
        )
    )


if __name__ == "__main__":
    main()
