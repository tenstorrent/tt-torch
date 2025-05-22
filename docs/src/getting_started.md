# Getting Started

This document walks you through how to set up to run models using tt-torch. The following topics are covered: 

* Configuring Hardware
* Setting up the Docker Container
* Installing Dependencies
* Creating a Virtual Environment
* Installing a Wheel
* Running a Model

## Configuring Hardware 

Configure your hardware with tt-installer: 

```bash
TT_SKIP_INSTALL_PODMAN=0 TT_SKIP_INSTALL_METALIUM_CONTAINER=0 /bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)"
```

## Setting up the Docker Container 

The simplest way to run models is to use the Docker image. You should have 50G free for the container.

Docker Image: This image includes all the necessary dependencies **ghcr.io/tenstorrent/tt-forge-fe/tt-forge-fe-base-ird-ubuntu-22-04**.

1. Install Docker if you do not already have it: 

```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```

2. Test that Docker is installed: 

```bash
docker --version
```

3. Add your user to the Docker group: 

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Creating a Virtual Environment

It is recommended that you install a virtual environment for the wheel you want to work with. Wheels from different repos may have conflicting dependencies.

Create a virtual environment (the environment name in the command is an example for the command, it's not required to use the same name listed):

```bash
python3 -m venv torch-venv
source torch-venv/bin/activate
```

## Installing tt-torch

### Torchvision Install (Required if You Need to Install Torchvision)

**If you intend to use torchvision in your project then this step must be done before installing the tt-torch wheel**

You must build the torchvision wheel yourself with certain build flags. This is because torchvision does not publish a wheel which uses the PyTorch CXX11 ABI.

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

### Installation Notes
- `tt-torch` requires a pytorch installation that ships with their ABI.
    - The `tt-torch` wheel lists the following version of torch as an installation requirement:
      `torch@https://download.pytorch.org/whl/cpu-cxx11-abi/torch-2.5.0%2Bcpu.cxx11.abi-cp310-cp310-linux_x86_64.whl`
    - This will be installed by pip upon installing the `tt-torch` wheel
- The `tt-torch` wheel contains a fork of `torch-mlir`. Please ensure that `torch-mlir` has not been installed in your venv before installing the `tt-torch` wheel.



### Installing the tt-torch Wheel

To install the tt-torch wheel do the following: 


1. Choose a `tt-torch` wheel from [here](https://github.com/tenstorrent/tt-forge/releases)

2. Install the wheel (this is an example that shows the link structure):
```bash
pip install https://github.com/tenstorrent/tt-forge/releases/download/nightly-0.1.0.dev20250519060217/tt_torch-0.1.0.dev20250519060217-cp310-cp310-linux_x86_64.whl
```

### Updating `PYTHONPATH`

In addition to the `tt-torch` python library that gets installed in `<YOUR_ENV_ROOT>/lib/python3.x/site-packages`, some binaries will be installed in `<YOUR_ENV_ROOT>/lib`, and some files from [tt-metal](https://github.com/tenstorrent/tt-metal) will be installed under `<YOUR_ENV_ROOT>/tt-metal`. Python needs to see these installations and so you should update your `PYTHONPATH` environment variable to include them:
```bash
export PYTHONPATH=$PYTHONPATH:<YOUR_ENV_ROOT>:<YOUR_ENV_ROOT>/lib
```

## Compiling and Running a Model

Once you have your `torch.nn.Module` compile the model:
```py
from tt_torch.dynamo.backend import backend
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        ...

    def foward(self, ...):
        ...

model = MyModel()

model = torch.compile(model, backend=backend)

inputs = ...

outputs = model(inputs)
```

## Example - Add Two Tensors

Here is an exampe of a small model which adds its inputs running through tt-torch. Try it out!

```py
from tt_torch.dynamo.backend import backend
import torch

class AddTensors(torch.nn.Module):
  def forward(self, x, y):
    return x + y


model = AddTensors()
tt_model = torch.compile(model, backend=backend)

x = torch.ones(5, 5)
y = torch.ones(5, 5)
print(tt_model(x, y))
```
