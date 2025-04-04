
## Prerequisites:

Main project dependencies are:
 - clang 17
 - Ninja
 - CMake >= 3.30
 - git LFS
 - python 0

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

This project requires the **GCC 11 toolchain**.
To check which GCC toolchain is currently in use, run:
```bash
clang -v
```
Look for the line that starts with: `Selected GCC installation:`. If it is something other than GCC 11, please uninstall that and install GCC 11 using:
```bash
sudo apt-get install gcc-11 lib32stdc++-11-dev lib32gcc-11-dev
```

The project also requires a toolchain build. By default, the toolchain is built in `/opt/ttmlir-toolchain`. This path is controlled by the `TTMLIR_TOOLCHAIN_DIR` environment variable.

The toolchain installation only needs to be done once, by running the following commands:

```bash
# Create toolchain dir
sudo mkdir -p /opt/ttmlir-toolchain
sudo chown -R $USER /opt/ttmlir-toolchain


# Build environment
cd third_party
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/
cmake -B toolchain -DBUILD_TOOLCHAIN=ON
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
