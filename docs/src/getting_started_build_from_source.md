


----OLD------
# Getting Started

This document walks you through how to build from source so you can develop the TT-Torch frontend. The following topics are covered:

* [Building TT-Torch From Source](#building-tt-torch-from-source)
    * [System Dependencies](#system-dependencies)
    * [Configuring Hardware](#configuring-hardware)
    * [Installing Dependencies](#installing-dependencies)
    * [Installing CMake](#installing-cmake-402)
    * [Installing Clang 17](#installing-clang-17)
    * [How to Build From Source](#how-to-build-from-source)
* [Next Steps](#next-steps)

## System Dependencies

TT-Torch has the following system dependencies:
* Ubuntu 22.04
* Python 3.10
* python3.10-venv
* Clang 17
* GCC 11
* Ninja
* CMake 3.20 or higher

```bash
sudo apt install clang cmake ninja-build pip python3.10-venv
```

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

### Installing Dependencies

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
    lcov
```

Install OpenMPI:

```bash
sudo wget -q https://github.com/dmakoviichuk-tt/mpi-ulfm/releases/download/v5.0.7-ulfm/openmpi-ulfm_5.0.7-1_amd64.deb -O /tmp/openmpi-ulfm.deb && sudo apt install /tmp/openmpi-ulfm.deb
```

### Installing CMake 4.0.2

Install CMake 4.0.2:

```bash
pip install cmake
```

### Installing Clang 17
This section walks you through installing Clang 17.

1. Install Clang 17:

```bash
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo ./llvm.sh 17
sudo apt install -y libc++-17-dev libc++abi-17-dev
sudo ln -s /usr/bin/clang-17 /usr/bin/clang
sudo ln -s /usr/bin/clang++-17 /usr/bin/clang++
```

2. Check that the selected GCC candidate using Clang 17 is using 11:

```bash
clang -v
```

Look for the line that starts with: `Selected GCC installation:`. If it is something other than GCC 11, please uninstall that and install GCC 11 using:

```bash
sudo apt-get install gcc-11 lib32stdc++-11-dev lib32gcc-11-dev
```

You **do not** need to uninstall other versions of GCC. Instead, you can use `update-alternatives` to configure the system to prefer GCC 11:

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
```

This approach lets multiple GCC versions coexist on your system and you can switch between them as needed.

3. Delete any non-11 paths:

```bash
sudo rm -rf /usr/bin/../lib/gcc/x86_64-linux-gnu/12
```

### How to Build From Source
This section describes how to build tt-torch. You need to build tt-torch whether you plan to do development work, or run models.

1. Clone the tt-torch repo:

```bash
git clone https://github.com/tenstorrent/tt-torch.git
cd tt-torch
```

2. Create a toolchain directory and make the account you are using the owner:

```bash
sudo mkdir -p /opt/ttmlir-toolchain
sudo chown -R $USER /opt/ttmlir-toolchain
```

3. Build the toolchain for tt-torch (this build step only needs to be done once):

```bash
cd third_party
cmake -B toolchain -DBUILD_TOOLCHAIN=ON
```

>**NOTE:** This step takes a long time to complete.

4. Navigate back to the **tt-torch** home directory.

5. Build TT-Torch:

```bash
source env/activate
cmake -G Ninja -B build
cmake --build build
cmake --install build
```

>**NOTE:** It takes a while for everything to build.

## Next Steps

Please follow [Testing](test.md)
