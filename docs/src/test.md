# Testing

## Infrastructure

tt-torch uses [pytest](https://docs.pytest.org/en/stable/) for all unit and model tests.

## PyTorch Unit Tests

We recommend running PyTorch unit tests before contributing to tt-torch. PyTorch unit tests are located under `tests/torch`.

### Single Device Tests

Below is a table explaining PyTorch unit tests for any device.

| Test | When to Run |
|------|-------------|
| `tests/torch/test_basic.py` | This test provides coverage for all the basic torch ops and functionality our compiler supports. Please run this test as soon as you setup tt-torch. You should see all tests passed. If any failed, you need to resolve setup problems. |
| `tests/torch/test_basic_multiple_execution.py` | This test provides coverage for running a compiled model multiple times. |
| `tests/torch/test_compare.py` | This test provides coverage for running comparison ops (eq, gt, l, etc) |
| `tests/torch/test_constant_fold.py` | This test provides coverage for running constant evaluation. |
| `tests/torch/test_conv2d.py` | This test provides coverage for 2d convolution ops. |
| `tests/torch/test_device_manager.py` | This test provides coverage for tt-mlir device management apis. |
| `tests/torch/test_interpolation.py` | This test provides coverage for upsample/ interpolation ops. |
| `tests/torch/test_logical.py` | This test provides coverage for logical ops (not, and, etc) |
| `tests/torch/test_maxpool2d.py` | This test provides coverage for maxpool 2d op. |
| `tests/torch/test_reduction.py` | This test provides coverage for reduction ops (sum, reshape, amin, etc) |
| `tests/torch/test_softmax.py` | This test provides coverage for softmax op. |

**How to run:** `pytest -svv <test name>`

**Example:** `pytest -svv tests/torch/test_basic.py`

### Multiple Device Tests

Below is a table explaining PyTorch unit tests for multiple device setup.

| Test | When to Run |
|------|-------------|
| `tests/torch/test_basic_async.py` | This test provides coverage for all the basic torch ops and functionality our compiler supports on all detected Tenstorrent devices, asynchronously and independently. Please run this test as soon as you setup tt-torch. You should see all tests passed. If any failed, you need to resolve setup problems. |
| `tests/torch/test_basic_multichip.py` | This test provides coverage for PyTorch `device_map` functionality. |

**How to run:** `pytest -svv <test name>`

**Example:** `pytest -svv tests/torch/test_basic_async.py`

## Onnx Unit Tests

We only have one test to check basic onnx ops implementation: `pytest -svv tests/onnx/test_basic.py`

## Model Tests

You can view our model zoo under `tests/models`.

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
