# How to add model tests?

## Requirements
[Build your environment](https://docs.tenstorrent.com/tt-torch/build.html)

## TT-Torch Backend in a nutshell

### `ModelTester` and `OnnxModelTester`
Our testing framework uses `ModelTester`, `OnnxModelTester` defined under [tests/utils.py](https://github.com/tenstorrent/tt-torch/blob/main/tests/utils.py)
`ModelTester` and `OnnxModelTester` are designed to facilitate the testing of PyTorch and ONNX models, respectively. These classes provide a structured framework for loading models, preparing inputs, running inference, and verifying the accuracy of the outputs.

#### `ModelTester`
The `ModelTester` class serves as a base class for testing PyTorch models. It handles common testing procedures and provides abstract methods that derived classes can implement for specific model loading and input preparation.
Derived classes must implement the following abstract methods:

-   `_load_model()`: This method should load the PyTorch model to be tested and return the model object.
-   `_load_inputs()`: This method should load or generate the input data for the model and return it. The input should be a Torch object.
-   `_extract_outputs()` (optional): This method should return a tuple of torch tensors based on the outputs if `ModelTester` `_extract_outputs` fails.

#### `OnnxModelTester`
The `OnnxModelTester` class inherits from `ModelTester` and extends it to specifically handle testing of ONNX models.

Derived classes must implement the following abstract methods:

-   `_load_model()`: This method should load the Onnx model to be tested and return the model object.
-   `_load_inputs()`: This method should load or generate the input data for the model and return it. The input should be a Torch object.
-   `_extract_outputs()` (optional): This method should return a tuple of torch tensors based on the outputs if `ModelTester` `_extract_outputs` fails.

### Backend
Backends are described under [tt_torch/dynamo/backend.py](https://github.com/tenstorrent/tt-torch/blob/main/tt_torch/dynamo/backend.py) and [tt_torch/onnx_compile/onnx_compile.py](https://github.com/tenstorrent/tt-torch/blob/main/tt_torch/onnx_compile/onnx_compile.py)
There are a few factors determining which backend to use:
```
class CompileDepth(Enum):
    TORCH_FX = 1
    STABLEHLO = 2
    TTNN_IR = 3
    COMPILE_OP_BY_OP = 4
    EXECUTE_OP_BY_OP = 5
    EXECUTE = 6
```
```
class OpByOpBackend(Enum):
    TORCH = 1
    STABLEHLO = 2
```

#### Backends for Torch Models:
- Op by Op Flows (`COMPILE_OP_BY_OP`/ `EXECUTE_OP_BY_OP`):
    - `OpByOpBackend` = `TORCH` --> uses `TorchExecutor`
    - `OpByOpBackend` = `STABLEHLO` --> uses `StablehloExecutor`
- Other Compile Depths:
    - Only `OpByOpBackend` = `TORCH` is allowed.
    - Uses `Executor`

#### Backends for ONNX Models:
- Op by Op Flows (`COMPILE_OP_BY_OP`/ `EXECUTE_OP_BY_OP`):
    Only `OpByOpBackend` = `STABLEHLO` is allowed.
    Uses `StablehloExecutor`
- Other Compile Depths:
    Only `OpByOpBackend` = `STABLEHLO` is allowed.
    Uses `OnnxExecutor`

### Executor
TT-Torch provides a set of executor classes that handle different types of models (ONNX, PyTorch) and compilation strategies (full compilation, op-by-op, etc.). The executor classes form a hierarchy, with specialized executors for different scenarios.

```
Executor (Base)
├── OpByOpExecutor
│   ├── TorchExecutor
│   └── StablehloExecutor
└── OnnxExecutor
```

- `Executor`, `OnnxExecutor` and `OpByOpExecutor` are defined under [tt_torch/dynamo/executor.py](https://github.com/tenstorrent/tt-torch/blob/main/tt_torch/dynamo/executor.py)
- `TorchExecutor` is defined under [tt_torch/dynamo/torch_backend.py](https://github.com/tenstorrent/tt-torch/blob/main/tt_torch/dynamo/torch_backend.py)
- `StablehloExecutor` is defined under [tt_torch/dynamo/shlo_backend.py](https://github.com/tenstorrent/tt-torch/blob/main/tt_torch/dynamo/shlo_backend.py)

#### Executor (Base Class)
The Executor class is the foundation for all executor implementations. It provides the basic framework for:

- Managing model representations (PyTorch programs, etc.)
- Converting input types between different formats
- Handling constants and model parameters
- Executing compiled models via TT-MLIR
- Managing device resources
- Verifying execution results

##### Key methods:

- `__call__`: Main entry point for executing the model
- `set_binary`: Sets the compiled binary for execution
- `typecast_inputs`: Converts inputs to hardware-supported types
- `register_intermediate_callback`: Sets up callbacks for runtime verification

#### OpByOpExecutor
OpByOpExecutor extends the base Executor to support operation-by-operation compilation and execution. This allows for:

- Detailed profiling of individual operations
- Verification of each operation's outputs
- Debugging specific operations that might fail

##### Key methods:

- `compile_op`: Compiles a single operation
- `run_op`: Executes a single compiled operation

#### TorchExecutor
TorchExecutor is specialized for handling PyTorch models in an op-by-op fashion. It:

- Processes PyTorch FX graph modules node by node
- Converts PyTorch operations to StableHLO
- Compares outputs with golden (PyTorch) outputs for verification

##### Key methods:

- `get_stable_hlo_graph`: Converts a PyTorch operation to StableHLO IR
- `run_gm_op_by_op`: Executes a graph module operation by operation

#### StablehloExecutor
StablehloExecutor specializes in executing models through the StableHLO IR. It can:

- Process ONNX models converted to StableHLO
- Process PyTorch models converted to StableHLO
- Execute individual StableHLO operations

##### Key methods:

- `add_program`: Adds a PyTorch program to the executor
- `add_onnx_model_proto`: Adds an ONNX model to the executor
- `get_stable_hlo_graph`: Prepares a StableHLO operation for compilation
- `shlo_op_by_op`: Executes StableHLO operations individually

#### OnnxExecutor
OnnxExecutor is designed for handling ONNX models. It can:

- Execute ONNX models using ONNX Runtime
- Execute ONNX models converted to TT-MLIR binaries

### CompilerConfig

This class manages settings for running models on Tenstorrent devices. Key aspects include:

* Compilation Depth: Defines the level of the compilation pipeline to reach.
* Profiling: Enables the collection of performance data for individual operations.
* Verification: Controls various checks and validations during compilation.
* Environment Overrides: Allows configuration through environment variables. This is explained in detail under [Controlling Compiler Behaviour](https://docs.tenstorrent.com/tt-torch/controlling.html)

Please see [tt_torch/tools/utils.py](https://github.com/tenstorrent/tt-torch/blob/main/tt_torch/tools/utils.py) for detailed information.

## How to write a test?
The following is an example test body:
```
# Insert SPDX licensing. Pre-commit will insert if it is missing
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# some base imports that are required for all tests:
import torch
import pytest
import onnx # for Onnx Tests

from tests.utils import ModelTester # for PyTorch Tests
from tests.utils import OnnxModelTester # for Onnx Tests
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend

class ThisTester(ModelTester): # or class ThisTester(OnnxModelTester):
    def _load_model(self):
        model = ....
        return model
    def _load_inputs(self):
        inputs = ...
        return inputs

# you can pytest parameterize certain arguments. i.e. Mode, OpByOpBackend, Model Name
@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "model_name_0",
        "model_name_1",
    ],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
# For PyTorch Tests
def <test_name>(record_property, model_name, mode, op_by_op):

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()

    if mode == "eval":
        # code to evaluate the output is as expected
    tester.finalize()

# For Onnx Tests:
def <test_name>(record_property, model_name, mode, op_by_op):
    cc = CompilerConfig()
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group="red",
    )

    results = tester.test_model()
    if mode == "eval":
        # code to evaluate the output is as expected
    tester.finalize()
```
You can find example tests under [tests/models](https://github.com/tenstorrent/tt-torch/tree/main/tests/models)
Note: please make sure to distinguish Onnx tests by appending `_onnx` to test names. i.e. `test_EfficientNet_onnx.py`
