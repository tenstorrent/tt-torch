# tt-torch

> **NOTE:** TT-Torch is deprecated. To work with PyTorch and the various features available in TT-Torch, please see the documentation for [TT-XLA](https://github.com/tenstorrent/tt-xla/blob/main/README.md). 

tt-torch is a [PyTorch2.0](https://pytorch.org/get-started/pytorch-2.0/) and [torch-mlir](https://github.com/llvm/torch-mlir/) based front-end for [tt-mlir](https://github.com/tenstorrent/tt-mlir/).

tt-torch uses venv to keep track of all dependencies. After [compiling](https://docs.tenstorrent.com/tt-torch/getting_started.html) you can activate the venv by running from the project root directory:

```bash
source env/activate
```

The currently supported models can be found [here](https://docs.tenstorrent.com/tt-torch/models/supported_models.html).
There is a brief demo showing how to use the compiler in *demos/resnet/resnet50_demo.py*

The general compile flow is:
 1. Pytorch model -> torch.compile which creates an fx graph
 2. Several compiler passes on the fx graph including consteval and dead code removal
 3. Conversion to torch-mlir -> torch-backend-mlir -> stableHLO through torch-mlir
 4. Conversion to TTIR -> TTNN -> flatbuffer through tt-mlir
 5. Creating executor with flatbuffer and passing back to user
 6. Copying inputs to device and executing flatbuffer through tt-mlir on each user invocation

In order to speed up model bring-up, users have the option of compiling models op-by-op. This allows in-parallel testing of the model since compilation does not stop at the first error. If enabled, see [Controlling Compilation](https://docs.tenstorrent.com/tt-torch/controlling.html), after step 2, compilation stops and the fx graph is passed to the executor which is returned to the user. Upon execution, whenever a new, unique op is seen (based on op-type and shape on inputs), a new fx graph is created with just one operation, inputs and outputs. This small graph then proceeds through steps 3-4 and is executed in place.

Results of each unique op execution are stored in a json file to be later parsed into either a spreadsheet, or uploaded to a database.

Op-by-op execution is currently performed on the pytorch fx graph, we'll be adding support for op-by-op on the stableHLO graph soon to allow op-by-op bringup of onnx models.

The repository uses pre-commit, read more about it [here](https://docs.tenstorrent.com/tt-torch/pre_-_commit.html).
