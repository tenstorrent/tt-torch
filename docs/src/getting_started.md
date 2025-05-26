# Getting Started

## System Dependencies

`tt-torch` requires the python 3.10 dev package, as well as the venv package. If not already installed, please run the following:

```bash
sudo apt-get install python3.10-dev python3.10-venv
```

## Creating a Virtual Environment (skip if you already have one)

Create a virtual environment if you do not already have one in your project:
```bash
python3.10 -m venv myvenv
```
This will create a virtual environemnt in the folder `myvenv` in the current directory.

Activate the environemnt:
```bash
source myvenv/bin/activate
```

## Installing tt-torch

### Installation Notes
- `tt-torch` requires a pytorch installation that ships with their ABI.
    - The `tt-torch` wheel lists the following version of torch as an installation requirement:
      `torch@https://download.pytorch.org/whl/cpu-cxx11-abi/torch-2.5.0%2Bcpu.cxx11.abi-cp310-cp310-linux_x86_64.whl`
    - This will be installed by pip upon installing the `tt-torch` wheel
- The `tt-torch` wheel contains a fork of `torch-mlir`. Please ensure that `torch-mlir` has not been installed in your venv before installing the `tt-torch` wheel.

### Torchvision Install (Required if you need to install torchvision)

**If you intend to use torchvision in your project then this step must be done before installing the tt-torch wheel**

You will need to build the torchvision wheel yourself with certain build flags. This is because torchvision does not publish a wheel which uses the PyTorch CXX11 ABI.

To install torchvision:
```bash
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.20.0 # tt-torch requires PyTorch 2.5.0. torchvision 0.20 is the latest version of torchvision that is compatible with PyTorch 2.5.0
pip uninstall -y torchvision # Ensure torchvision is not in your virtual environment
pip install wheel
pip install torch@https://download.pytorch.org/whl/cpu-cxx11-abi/torch-2.5.0%2Bcpu.cxx11.abi-cp310-cp310-linux_x86_64.whl
TORCHVISION_USE_VIDEO_CODEC=0 TORCHVISION_USE_FFMPEG=0 _GLIBCXX_USE_CXX11_ABI=1 USE_CUDA=OFF python setup.py bdist_wheel
pip install dist/torchvision*.whl --force-reinstall
```

If the install was successful then there's no need to keep the torchvision source around:
```bash
cd ..
rm -rf vision
```

### Installing the tt-torch wheel

Download a `tt-torch` wheel from [here](https://github.com/tenstorrent/tt-forge/releases)

Install the wheel:
```bash
pip install <PATH_TO_TT_TORCH_WHEEL>.whl
```

### Updating `PYTHONPATH`

In addition to the `tt-torch` python library that gets installed in `<YOUR_ENV_ROOT>/lib/python3.x/site-packages`, some binaries will be installed in `<YOUR_ENV_ROOT>/lib`, and some files from [tt-metal](https://github.com/tenstorrent/tt-metal) will be installed under `<YOUR_ENV_ROOT>/tt-metal`. Python needs to see these installations and so you should update your `PYTHONPATH` environment variable to include them:
```bash
export PYTHONPATH=$PYTHONPATH:<YOUR_ENV_ROOT>:<YOUR_ENV_ROOT>/lib
```

## Compiling and Running a Model

Once you have your `torch.nn.Module` compile the model:
```py
from tt_torch.dynamo.backend import backend, BackendOptions
from tt_torch.tools.device_manager import DeviceManager

import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        ...

    def foward(self, ...):
        ...

options = BackendOptions()
options.devices = [device]

model = MyModel()

model = torch.compile(model, backend=backend, options=options)

inputs = ...

outputs = model(inputs)
```

## Example - Add Two Tensors

Here is an exampe of a small model which adds its inputs running through tt-torch. Try it out!

```py
from tt_torch.dynamo.backend import backend, BackendOptions
from tt_torch.tools.device_manager import DeviceManager
import torch

class AddTensors(torch.nn.Module):
  def forward(self, x, y):
    return x + y


device = DeviceManager.create_parent_mesh_device([1, 1])

options = BackendOptions()
options.devices = [device]

model = AddTensors()
tt_model = torch.compile(model, backend=backend, options=options)

x = torch.ones(5, 5)
y = torch.ones(5, 5)
print(tt_model(x, y))
DeviceManager.release_parent_device(device)
```
