# Getting Started

This document describes how to build the TT-Torch project on your local machine. You must build from source if you want to develop for TT-Torch. If you only want to run models, please choose one of the following sets of instructions instead:
* [Installing a Wheel and Running an Example](getting_started.md) - You should choose this option if you want to run models.
* [Using a Docker Container to Run an Example](getting_started_docker.md) - Choose this option if you want to keep the environment separate from your existing environment.

The following topics are covered:

* [Configuring Hardware](#configuring-hardware)
* [System Dependencies](#system-dependencies)
* [Installing Dependencies](#installing-dependencies)
* [Running a Test Model](#running-a-test-model)

> **NOTE:** If you encounter issues, please request assistance on the
>[TT-Torch Issues](https://github.com/tenstorrent/tt-torch/issues) page.

## Configuring Hardware
Before setup can happen, you must configure your hardware. You can skip this section if you already completed the configuration steps. Otherwise, this section of the walkthrough shows you how to do a quick setup using TT-Installer.

1. Configure your hardware with TT-Installer using the [Quick Installation section here.](https://docs.tenstorrent.com/getting-started/README.html#quick-installation)

2. Reboot your machine.

3. Please ensure that after you run this script, after you complete reboot, you activate the virtual environment it sets up - ```source ~/.tenstorrent-venv/bin/activate```.

4. After your environment is running, to check that everything is configured, type the following:

```bash
tt-smi
```

You should see the Tenstorrent System Management Interface. It allows you to view real-time stats, diagnostics, and health info about your Tenstorrent device.

![TT-SMI](./imgs/tt_smi.png)

## System Dependencies

TT-Torch has the following system dependencies:
* Ubuntu 22.04
* Python 3.10
* python3.10-venv
* Clang 17
* GCC 11
* Ninja
* CMake 3.20 or higher

### Installing Python
If your system already has Python installed, make sure it is Python 3.10 or higher:

```bash
python3 --version
```

If not, install Python:

```bash
sudo apt install python3
```

### Installing CMake 4.0.3
This section walks you through installing CMake 4 or higher.

1. Install CMake 4.0.3:

```bash
pip install cmake==4.0.3
```

2. Check that the correct version of CMake is installed:

```bash
cmake --version
```

If you see ```cmake version 4.0.3``` you are ready for the next section.

### Installing LLVM 17 Toolchain
This section walks you through installing LLVM 17 Toolchain.

1. Download LLVM 17 Toolchain Installation Script:

```bash
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo ./llvm.sh 17
```

2. Install C++ Standard Library (libc++) Development Files for LLVM 17:

```bash
sudo apt install -y libc++-17-dev libc++abi-17-dev
```

3. Symlink Clang:

```bash
sudo ln -s /usr/bin/clang-17 /usr/bin/clang
sudo ln -s /usr/bin/clang++-17 /usr/bin/clang++
```

4. Check that the selected GCC candidate using Clang 17 is using 11:

```bash
clang -v
```

5. Look for the line that starts with: `Selected GCC installation:`. If it is something other than GCC 11, and you do not see GCC 11 listed as an option, please install GCC 11 using:

```bash
sudo apt-get install gcc-11 lib32stdc++-11-dev lib32gcc-11-dev
```

6. If you see GCC 12 listed as installed and listed as the default choice, uninstall it with:

```bash
sudo rm -rf /usr/bin/../lib/gcc/x86_64-linux-gnu/12
```

## Installing Dependencies

Install additional dependencies required by TT-Torch that were not installed by the TT-Installer script:

```bash
sudo apt-get install -y \
    libhwloc-dev \
    libtbb-dev \
    libcapstone-dev \
    pkg-config \
    linux-tools-generic \
    ninja-build \
    libgtest-dev \
    ccache \
    doxygen \
    graphviz \
    patchelf \
    libyaml-cpp-dev \
    libboost-all-dev \
    lcov \
    protobuf-compiler
```

Install OpenMPI:

```bash
sudo wget -q https://github.com/dmakoviichuk-tt/mpi-ulfm/releases/download/v5.0.7-ulfm/openmpi-ulfm_5.0.7-1_amd64.deb -O /tmp/openmpi-ulfm.deb && sudo apt install /tmp/openmpi-ulfm.deb
```

>**NOTE:** If you want to run a test which uses `cv2` (i.e. YOLOv10), please install libgl.


### How to Build From Source
This section describes how to build TT-Torch. You need to build TT-Torch whether you plan to do development work, or run models.

1. Clone the TT-Torch repo:

```bash
git clone https://github.com/tenstorrent/tt-torch.git
cd tt-torch
```

2. Create a toolchain directory and make the account you are using the owner:

```bash
sudo mkdir -p /opt/ttmlir-toolchain
sudo chown -R $USER /opt/ttmlir-toolchain
```

3. Build the toolchain for TT-Torch (this build step only needs to be done once):

```bash
cd third_party
cmake -B toolchain -DBUILD_TOOLCHAIN=ON
cd -
```

>**NOTE:** This step takes a long time to complete.

4. Build TT-Torch:

```bash
source env/activate
cmake -G Ninja -B build
cmake --build build
cmake --install build
```

>**NOTE:** It takes a while for everything to build.

After the build completes, you are ready to run a test model.

## Running a Test Model
You can test your installation by running the **resnet50_demo.py**:

```bash
python demos/resnet/resnet50_demo.py
```

For additional tests and models, please follow [Testing](test.md).
