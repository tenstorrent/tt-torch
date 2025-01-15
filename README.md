# tt-torch

tt-torch is a [PyTorch2.0](https://pytorch.org/get-started/pytorch-2.0/) and [torch-mlir](https://github.com/llvm/torch-mlir/) based front-end for [tt-mlir](https://github.com/tenstorrent/tt-mlir/).


## Prerequisites:

Main project dependencies are:
 - clang 17
 - Ninja
 - CMake >= 3.30
 - git LFS
 - python 3.11

On Ubuntu 22.04 systems these can be installed using the following commands:

```bash
# Update package list
sudo apt update -y
sudo apt upgrade -y

# Install Clang
sudo apt install clang-17

# Install Ninja
sudo apt install ninja-build

# Install Git LFS
sudo apt install git-lfs

# Install CMake
sudo apt remove cmake -y
pip3 install cmake --upgrade
```

Ensure cmake can by found in this path pip installed it to. E.g. by adding `PATH=$PATH:$HOME/.local/bin` to your `.bashrc` file, and verify installation:
```bash
cmake --version
```

The project also requires the toolchain built from tt-mlir. This is currently a manual step (to be automated in the future). By default, the toolchain is built in `/opt/ttmlir-toolchain`. This path is controlled by the `TTMLIR_TOOLCHAIN_DIR` environment variable.

The toolchain installation only needs to be done once, by running the following commands:

```bash
# Create toolchain dir
sudo mkdir -p /opt/ttmlir-toolchain
sudo chown -R $USER /opt/ttmlir-toolchain

# Clone tt-mlir
git clone git@github.com:tenstorrent/tt-mlir.git
cd tt-mlir

# Build environment
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/
cmake -B env/build env
cmake --build env/build
cd -
```
For more information see [tt-mlir build steps](https://docs.tenstorrent.com/tt-mlir/build.html).

## Compile Steps:

Run the following commands to compile:
```bash
source env/activate
cmake -G Ninja -B build
cmake --build build
cmake --install build
```

Run a basic test to verify:
```bash
pytest tests/torch/test_basic.py
```

## Pre-Commit
Pre-Commit applies a Git hook to the local repository, ensuring linting is checked and applied on every git commit action. Install it from the root of the repository using:

```bash
source env/activate
pre-commit install
```

If you have already made commits before installing the pre-commit hooks, you can run the following to “catch up”:

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
| TT_TORCH_ENABLE_IR_PRINTING | Enables printing MLIR for all conversion steps from StableHLO to TTNN. Be warned, this forces single core compile, so is much slower. | False |
