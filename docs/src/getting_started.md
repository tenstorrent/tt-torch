# Getting Started

## System Dependencies

`tt-torch` requires the python 3.11 dev package, as well as the venv package. If not already installed, please run the following:

```bash
sudo apt-get install python3.11-dev python3.11-venv
```

## Creating a Virtual Environment (skip if you already have one)

Create a virtual environment if you do not already have one in your project:
```bash
python3.11 -m venv myvenv
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
      `torch@https://download.pytorch.org/whl/cpu-cxx11-abi/torch-2.5.0%2Bcpu.cxx11.abi-cp311-cp311-linux_x86_64.whl`
    - This will be installed by pip upon installing the `tt-torch` wheel
- The `tt-torch` wheel contains a fork of `torch-mlir`. Please ensure that `torch-mlir` has not been installed in your venv before installing the `tt-torch` wheel.

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
