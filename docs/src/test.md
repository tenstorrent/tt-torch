# Testing

This document walks you through how to test tt-torch after building from source or installing from wheel

* [Infrastructure](#infrastructure)
* [Unit Tests](#unit-tests)
* [Running the resnet Demo](#running-the-resnet-demo)
* [Compiling and Running a Model](#compiling-and-running-a-model)
* [Model Zoo](#model-zoo)

## Infrastructure

tt-torch uses [pytest](https://docs.pytest.org/en/stable/) for all unit and model tests.

## Unit Tests

- PyTorch unit tests are located under [tests/torch](https://github.com/tenstorrent/tt-torch/tree/main/tests/torch)
- Onnx unit tests are located under [tests/onnx](https://github.com/tenstorrent/tt-torch/tree/main/tests/onnx)

You can check that everything is working with a basic unit test:

```bash
# If building tt-torch from source
pytest -svv tests/torch/test_basic.py

# If installing tt-torch from wheel
pip install pytest
curl -s https://raw.githubusercontent.com/tenstorrent/tt-torch/main/tests/torch/test_basic.py -o test_basic.py
pytest -svv test_basic.py
```

>**NOTE:** If you are using a multiple device setup, we encourage you to run the following tests:
> - `tests/torch/test_basic_async.py`
> - `tests/torch/test_basic_multichip.py`

## Running the resnet Demo
You can also try a demo:

```bash
# If building tt-torch from source
python demos/resnet/resnet50_demo.py

# If installing tt-torch from wheel
pip install pytest
curl -s https://raw.githubusercontent.com/tenstorrent/tt-torch/main/demos/resnet/resnet50_demo.py -o resnet50_demo.py
python resnet50_demo.py
```

## Compiling and Running a Model

Once you have your `torch.nn.Module` compile the model:
```py
import torch
import tt_torch

class MyModel(torch.nn.Module):
    def __init__(self):
        ...

    def forward(self, ...):
        ...

model = MyModel()

model = torch.compile(model, backend="tt")

inputs = ...

outputs = model(inputs)
```

## Example - Add Two Tensors

Here is an example of a small model which adds its inputs running through tt-torch. Try it out!

```py
import torch
import tt_torch

class AddTensors(torch.nn.Module):
  def forward(self, x, y):
    return x + y


model = AddTensors()
tt_model = torch.compile(model, backend="tt")

x = torch.ones(5, 5)
y = torch.ones(5, 5)
print(tt_model(x, y))
```

# Testing With Experimental Flow

We are experimenting with `torch-xla` as a method of capturing and executing PyTorch models. We plan to eventually use torch-xla as the main execution engine for tt-torch.

### Experimental `torch.compile` Backend ("tt-experimental")

```py
import torch
import tt_torch

class AddTensors(torch.nn.Module):
  def forward(self, x, y):
    return x + y


model = AddTensors()
tt_model = torch.compile(model, backend="tt-experimental")

x = torch.ones(5, 5)
y = torch.ones(5, 5)
print(tt_model(x, y))
```

### Experimental Eager Execution

`torch-xla` allows us to execute PyTorch models eagerly using the `.to(device)` infrastructure.

```py
import torch
import tt_torch

class AddTensors(torch.nn.Module):
  def forward(self, x, y):
    return x + y


model = AddTensors()
model = model.to("xla")

x = torch.ones(5, 5, device="xla")
y = torch.ones(5, 5, device="xla")
print(model(x, y).to("cpu"))
```

## Model Zoo

You can view our model zoo under [tests/models](https://github.com/tenstorrent/tt-torch/tree/main/tests/models)

Please see [overview](https://docs.tenstorrent.com/tt-torch/controlling.html) for an explanation on how to control model tests.

You can always run `pytest --collect-only` to view available pytest names under test file.

```
/userhome/tt-torch$ pytest --collect-only tests/models/resnet
====================================================================== test session starts =======================================================================
platform linux -- Python 3.10.12, pytest-8.4.0, pluggy-1.6.0
rootdir: /userhome/tt-torch
configfile: pyproject.toml
plugins: cov-6.2.1
collected 12 items

<Dir tt-torch>
  <Package tests>
    <Dir models>
      <Dir resnet>
        <Module test_resnet.py>
          <Function test_resnet[single_device-op_by_op_stablehlo-train]>
          <Function test_resnet[single_device-op_by_op_stablehlo-eval]>
          <Function test_resnet[single_device-op_by_op_torch-train]>
          <Function test_resnet[single_device-op_by_op_torch-eval]>
          <Function test_resnet[single_device-full-train]>
          <Function test_resnet[single_device-full-eval]>
          <Function test_resnet[data_parallel-op_by_op_stablehlo-train]>
          <Function test_resnet[data_parallel-op_by_op_stablehlo-eval]>
          <Function test_resnet[data_parallel-op_by_op_torch-train]>
          <Function test_resnet[data_parallel-op_by_op_torch-eval]>
          <Function test_resnet[data_parallel-full-train]>
          <Function test_resnet[data_parallel-full-eval]>
```
