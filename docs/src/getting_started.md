# Getting Started

This document walks you through how to set up to run models using tt-torch. The following topics are covered:

* [System Dependencies](#system-dependencies)
* [Configuring Hardware](#configuring-hardware)
* [Installing Dependencies](#installing-dependencies)
* [Installing CMake](#installing-cmake-402)
* [Installing Clang 17](#installing-clang-17)
* [Building tt-torch](#building-tt-torch)
* [Test the tt-torch Build](#test-the-tt-torch-build)
* [Running the resnet Demo](#running-the-resnet-demo)
* [Compiling and Running a Model](#compiling-and-running-a-model)
* [Example - Add Two Tensors](#example---add-two-tensors)

## System Dependencies

tt-torch has the following system dependencies:
* Ubuntu 22.04
* Python 3.10
* python3.10-venv
* Clang 17
* GCC 11
* Ninja
* CMake 3.20 or higher

```bash
sudo apt install clang cmake ninja-build pip python3.10-venv
```

## Configuring Hardware

This walkthrough assumes you are using Ubuntu 22.04.

Configure your hardware with tt-installer:

1. Make sure your system is up-to-date:

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

2. Set up your hardware and dependencies using tt-installer:

```bash
/bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)"
```

## Installing Dependencies

Install additional dependencies that were not installed by the tt-installer script:

```bash
sudo apt-get install -y \
    libhwloc-dev \
    libtbb-dev \
    libcapstone-dev \
    pkg-config \
    linux-tools-generic \
    ninja-build \
    libgtest-dev \
    ccache \
    doxygen \
    graphviz \
    patchelf \
    libyaml-cpp-dev \
    libboost-all-dev \
    lcov
```

Install OpenMPI:

```bash
sudo wget -q https://github.com/dmakoviichuk-tt/mpi-ulfm/releases/download/v5.0.7-ulfm/openmpi-ulfm_5.0.7-1_amd64.deb -O /tmp/openmpi-ulfm.deb && sudo apt install /tmp/openmpi-ulfm.deb
```

## Installing CMake 4.0.2

Install CMake 4.0.2:

```bash
pip install cmake
```

## Installing Clang 17
This section walks you through installing Clang 17.

1. Install Clang 17:

```bash
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo ./llvm.sh 17
sudo apt install -y libc++-17-dev libc++abi-17-dev
sudo ln -s /usr/bin/clang-17 /usr/bin/clang
sudo ln -s /usr/bin/clang++-17 /usr/bin/clang++
```

2. Check that the selected GCC candidate using Clang 17 is using 11:

```bash
clang -v
```

Look for the line that starts with: `Selected GCC installation:`. If it is something other than GCC 11, please uninstall that and install GCC 11 using:

```bash
sudo apt-get install gcc-11 lib32stdc++-11-dev lib32gcc-11-dev
```

You **do not** need to uninstall other versions of GCC. Instead, you can use `update-alternatives` to configure the system to prefer GCC 11:

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
```

This approach lets multiple GCC versions coexist on your system and you can switch between them as needed.

3. Delete any non-11 paths:

```bash
sudo rm -rf /usr/bin/../lib/gcc/x86_64-linux-gnu/12
```

## Building tt-torch
This section describes how to build tt-torch. You need to build tt-torch whether you plan to do development work, or run models.

1. Clone the tt-torch repo:

```bash
git clone https://github.com/tenstorrent/tt-torch.git
cd tt-torch
```

2. Create a toolchain directory and make the account you are using the owner:

```bash
sudo mkdir -p /opt/ttmlir-toolchain
sudo chown -R $USER /opt/ttmlir-toolchain
```

3. Build the toolchain for tt-torch (this build step only needs to be done once):

```bash
cd third_party
cmake -B toolchain -DBUILD_TOOLCHAIN=ON
```

>**NOTE:** This step takes a long time to complete.

4. Navigate back to the **tt-torch** home directory.

5. Build tt-torch:

```bash
source env/activate
cmake -G Ninja -B build
cmake --build build
cmake --install build
```

>**NOTE:** It takes a while for everything to build.

## Test the tt-torch Build:
You can check that everything is working with a basic unit test:

```bash
pytest -svv tests/torch/test_basic.py
```

>**NOTE:** Any time you use tt-torch, you need to be in the activated virtual
> environment you created. Otherwise, you will get an error when trying to run
> a test.

## Running the resnet Demo
You can also try a demo:

```bash
python demos/resnet/resnet50_demo.py
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
