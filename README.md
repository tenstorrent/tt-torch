# tt-torch

The tt-torch project requires environment setup from tt-mlir project. https://github.com/tenstorrent/tt-mlir/
See documentation of tt-mlir and follow Environment setup. https://docs.tenstorrent.com/tt-mlir/build.html

## Compile Steps:
```
source env/activate
cmake -G Ninja -B build
cmake --build build
cmake --install build
```

## Pre-Commit
Pre-Commit applies a git hook to the local repository such that linting is checked and applied on every `git commit` action. Install from the root of the repository using:

```bash
source env/activate
pre-commit install
```

If you have already committed before installing the pre-commit hooks, you can run on all files to "catch up":

```bash
pre-commit run --all-files
```

For more information visit [pre-commit](https://pre-commit.com/)


## Controlling Behaviour

You can use the following environment variables to override default behaviour:

| Environment Variable | Behaviour | Default |
| -------------------- | --------- | --------
| TT_TORCH_COMPILE_DEPTH | Sets the maximum compile depth, see `tt_torch/tools/utils.py` for options. | `EXECUTE` |
| TT_TORCH_VERIFY_INTERMEDIATES | Sets whether to verify intermediate tensors against pytorch when running with compile depth `EXECUTE_OP_BY_OP`. | False |
| TT_TORCH_CONSTEVAL | Enables evaluation of constant expressions (consteval) in the Torch FX graph prior to compilation. | False |
| TT_TORCH_CONSTEVAL_PARAMETERS | Extends consteval to include parameters (e.g., model weights) as well as embedded constants. | False |
| TT_TORCH_EMBEDDEDD_CONSTANTS | Remove embedded constants from the Torch FX graph and convert them to constant inputs | False |
| TT_TORCH_IR_LOG_LEVEL | Enables printing MLIR from Torch to TTNN. It supports two modes; `INFO` and `DEBUG`. `INFO` prints MLIR for all conversions steps (Torch, StableHLO, TTIR and TTNN MLIR graphs). `DEBUG` prints intermediate MLIR for all passes (IR dump before and after each pass) additionally. Be warned, `DEBUG` IR printing forces single core compile, so it is much slower. | Disable |
