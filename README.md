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
| TT_TORCH_VERIFY_INTERMEDIATES | Sets whether to verify intermediate tensors agains pytorch when running with compile depth `EXECUTE_OP_BY_OP`. | False |
| TT_TORCH_CONSTEVAL | Sets whether to enable consteval on the torch fx graph before compiling. | False |
| TT_TORCH_CONSTEVAL_PARAMETERS | Sets whether to also consteval the parameters, not only the embedded constants. | False |
| TT_TORCH_ENABLE_IR_PRINTING | Sets whether to enable printing MLIR for all conversion steps from StableHLO to TTNN. Be warned, this forces single core compile, so is much slower. | False |
