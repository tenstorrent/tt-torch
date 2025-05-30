
## Prerequisites:

Main project dependencies are:
 - clang 17
 - Ninja
 - CMake >= 3.30
 - python 3.10

On Ubuntu 22.04 systems these can be installed using the following commands:

```bash
# Update package list
sudo apt update -y
sudo apt upgrade -y

# Install Clang
sudo apt install clang-17

# Install Ninja
sudo apt install ninja-build

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
For more information see [tt-mlir getting started](https://docs.tenstorrent.com/tt-mlir/getting-started.html).

## Compile Steps:

Run the following commands to compile. Profiling builds require an extra step[^profiling_note]:
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

[^profiling_note]: For a profiling build, cmake build files should be generated with an extra directive, as `cmake -G Ninja -B build -DTT_RUNTIME_ENABLE_PERF_TRACE=ON`. Refer to [profiling docs](./profiling.md) for more information.
