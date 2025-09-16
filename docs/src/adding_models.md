# How to add model tests?

> **NOTE:** TT-Torch is deprecated. To work with PyTorch and the various features available in TT-Torch, please see the documentation for [TT-XLA](https://github.com/tenstorrent/tt-xla/blob/main/README.md).

## Requirements
[Getting Started](https://docs.tenstorrent.com/tt-torch/getting_started.html)

Following the instructions on this page will show you how to build your environment and run a demo.

## TT-Torch Backend in a Nutshell

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

## How to Write a Test?
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

## Test Run Nodes

- op-by-op flow: This will break down model into graphs and break down graphs into ops, compiling and executing unique (first seen occurrence) ops independently. Results are written to .json file and and optionally converted to XLS file for reporting, as post-processing step.  The op-by-op flow is typically used for bringing up new models and debugging and you should start there, especially if the model is a new, untested architecture or your have reason to believe it will not work end-to-end out of the box. Engaged with `cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP` in test, typically driven by pytest params `[op_by_op_torch-eval]`.

- full end-to-end flow: This is the typical compile + execute of the model that typically includes functional correctness checking. Engaged with `cc.compile_depth = CompileDepth.EXECUTE` in test, typically driven by pytest params `[full-eval]`.


## Where to Add Tests on tt-torch GitHub CI?

If you're a Tenstorrent internal developer and have a new model that is either running fully/correctly or still needs some work (compiler support, runtime support, etc), it should be added to CI in the same PR you add the model.  Below is guide for where to add it.


### Case 1: The New Model Test Runs Correctly End-to-end

If you've tried it and it runs – great!

- Add it to run in "nightly full model execute list" in `.github/workflows/run-full-model-execution-tests-nightly.yml` while ideally balancing existing groups of tests. Example:

```
tests/models/Qwen/test_qwen2_casual_lm.py::test_qwen2_casual_lm[full-eval]
```

- Also add it to "weekly op-by-op-flow list" in `.github/workflows/run-op-by-op-flow-tests-weekly.yml` where we less frequently run tests that have all ops passing through to `EXECUTE` depth in op-by-op flow. Example:

```
tests/models/Qwen/test_qwen2_casual_lm.py::test_qwen2_casual_lm[op_by_op_torch-eval]
```

### Case 2: The New Model Test Runs End-to-end but Encounters a PCC/ATOL/Checker Error

This is okay, there is still value in running the model.

- Follow previous section instructions for adding it to "nightly full model execute" and "weekly op-by-op-flow list" but first open a GitHub issue (follow template and `models_pcc_issue` label like the example below) to track the PCC/ATOL/Checker error, reference it in the test body so it can be tracked/debugged, and disable PCC/ATOL/Token checking as needed. Example:

```
# TODO Enable checking - https://github.com/tenstorrent/tt-torch/issues/490
assert_pcc=False,
assert_atol=False,
```

### Case 3: The New Model Test Does Not Run Correctly End-to-end

No problem. If your end-to-end model hits a compiler failure (unsupported op, etc) or runtime assert of any kind, this is why the op-by-op flow exists. The op-by-op flow is designed to flag per-op compile/runtime failures (which are perfectly fine) but is expected to return overall passed status.

- Go ahead and run the op-by-op flow locally (or on CI) for your model, and if the pytest finishes without fatal errors, add it to the "nightly op-by-op flow list" (a new or existing group) in `.github/workflows/run-op-by-op-flow-tests-nightly.yml` where individual ops will be tracked/debugged and later promoted to "nightly full model execute list" once ready. Example:

```
tests/models/t5/test_t5.py::test_t5[op_by_op_torch-t5-large-eval]
```

- It is helpful if you can run `python results/parse_op_by_op_results.py` (will generate `results/models_op_per_op.xlsx` for all models you've recently run in op-by-op-flow) and include the XLS file in your PR. This XLS file contains op-by-op-flow results and is also generated in Nightly regression for all work-in-progress models in `.github/workflows/run-op-by-op-flow-tests-nightly.yml`.

- If your model is reported in `results/models_op_per_op.xlsx` as being able to compile all ops successfully (ie. all ops can compile to status `6: CONVERTED_TO_TTNN`, but some hit runtime `7: EXECUTE` failures) then it should also be added to "nightly e2e compile list" in `.github/workflows/run-e2e-compile-tests.yml` which stops before executing the model via `TT_TORCH_COMPILE_DEPTH=TTNN_IR pytest ...`

## How to Load Test Files into/from Large File System (LFS)

We have set up access to a AWS S3 bucket to be able to load and access model related files for testing. We can load files into our S3 bucket and access them from the tester scripts. You will need access to S3 bucket portal to add files. **If you don't have an AWS account or access to the S3 bucket please reach out to the tt-torch community leader.** Then, depending on if the test is running on CI or locally we will be able to load the files from the CI/IRD LFS caches that automatically sync up with contents in S3 bucket.

### Load Files into S3 Bucket

Access S3 bucket portal, **if you don't have access to the S3 bucket please reach out to the tt-torch community leader**, and load file from local dir. Please add files following this structure:

```
test_files
├── pytorch
│   ├── huggingface
│   │   ├── meta-llama
│   │   │   ├── Llama-3.1-70B
│   │   │   │   └── <hugginface files>
│   │   │   ├── Llama-2-7b-hf
│   │   │   │   └── <hugginface files>
│   │   │   └── ...
│   │   └── ...
│   ├── yolov10
│   │   └── yolov10.pt
│   └── ...
└── onnx
    ├── ViT
    │   └── ViT.onnx
    └── ...
```

### Load Files from S3 Bucket

Once files is loaded into S3 bucket we can access the file using a helper function from the [tt-forge-models repo](https://github.com/tenstorrent/tt-forge-models/blob/main/tools/utils.py#L14):
```
def get_file(path):
```

```
from third_party.tt_forge_models.tools.utils import get_file

class ThisTester(ModelTester):
    def _load_model(self):
        file = get_file("test_files/pytorch/yoloyv10/yolov_10n.pt")

...

```

The `path` arg should be the full path of the file in the S3 bucket. **DO NOT use the S3 URL**

#### Loading Files Locally

Locally `get_file(path)` will pull files directly from an IRD LFS cache. The IRD LFS cache is set up to sync up with S3 bucket every 5-10 minutes. You will need to set the `IRD_LF_CACHE` environment variable to the appropriate address. **Contact tt-torch community leader for IRD LF cache address.**

File/s will be downloaded into a local cache so next time you want to access the same file we won't have to load from the IRD cache or download it again. The default location for the local cache is `~/.cache/lfcache/`. If you want to redirect files to a custom cache path set the `LOCAL_LF_CACHE` env variable to the desired path.

If you can't be granted access to the IRD LF cache you have two options:
1. We can use `path=url` to download files. **DO NOT use the S3 URL.**
2. We can use `path=<local_path>` where local path is a relative path of the file in your local cache (`~/.cache/lfcache/<local_path>` or `LOCAL_LF_CACHE/<local_path>`).

#### Loading Files from CI

Once a file has been loaded into ther S3 bucket the CI's shared `DOCKER_CACHE_DIR` has been set up to sync up with the contents of the S3 bucket every hour. `get_file()` will fetch the file from the `DOCKER_CACHE_DIR`.
