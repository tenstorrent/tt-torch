# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from skbuild import setup
import os
from skbuild.command.install_lib import install_lib
import glob
from setuptools import find_namespace_packages
import sys
import shutil


class install_metal_libs(install_lib):
    def run(self):
        install_lib.run(self)
        install_path = os.path.join(self.install_dir, "tt_mlir")
        os.makedirs(install_path, exist_ok=True)
        ttmlir_opt = os.path.abspath(
            os.path.join(
                os.getcwd(),
                "third_party",
                "tt-mlir",
                "src",
                "tt-mlir-build",
                "bin",
                "ttmlir-opt",
            )
        )
        self.copy_file(ttmlir_opt, install_path)

        # Copy profiling tools
        src_tools_dir = os.path.abspath(
            os.path.join(
                os.getcwd(),
                "third_party",
                "tt-mlir",
                "src",
                "tt-mlir",
                "third_party",
                "tt-metal",
                "src",
                "tt-metal",
                "build",
                "tools",
            )
        )
        dest_tools_dir = os.path.join(self.install_dir, "tt_metal", "tools")
        if os.path.exists(src_tools_dir):
            os.makedirs(os.path.dirname(dest_tools_dir), exist_ok=True)
            shutil.copytree(src_tools_dir, dest_tools_dir, dirs_exist_ok=True)

        if include_models:

            # Copy entire TT Forge Models repo (python)
            src_models_dir = os.path.abspath(
                os.path.join(os.getcwd(), "third_party", "tt_forge_models")
            )

            dest_models_dir = os.path.join(
                self.install_dir, "third_party", "tt_forge_models"
            )
            if os.path.exists(src_models_dir):
                os.makedirs(os.path.dirname(dest_models_dir), exist_ok=True)
                shutil.copytree(
                    src_models_dir,
                    dest_models_dir,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git"),
                )


# Compile time env vars
os.environ["DONT_OVERRIDE_INSTALL_PATH"] = "1"

cmake_args = [
    "-GNinja",
    "-DBUILD_TTRT=OFF",
]
if "--code_coverage" in sys.argv:
    cmake_args += [
        "-DCODE_COVERAGE=ON",
    ]
    sys.argv.remove("--code_coverage")

if "--build_perf" in sys.argv:
    cmake_args += [
        "-DTT_RUNTIME_ENABLE_PERF_TRACE=ON",
    ]
    sys.argv.remove("--build_perf")

if "--build_runtime_debug" in sys.argv:
    cmake_args += [
        "-DTT_RUNTIME_DEBUG=ON",
    ]
    sys.argv.remove("--build_runtime_debug")


# Include Models, for CI only - not for release.
include_models = False
if "--include-models" in sys.argv:
    include_models = True
    sys.argv.remove("--include-models")


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="tt_torch",
    version="0.1",
    author="Aleks Knezevic",
    author_email="aknezevic@tenstorrent.com",
    license="Apache-2.0",
    homepage="https://github.com/tenstorrent/tt-torch",
    packages=find_namespace_packages(include=["tt_torch*"])
    + find_namespace_packages(
        where="third_party/torch-mlir/src/torch-mlir-build/python_packages/torch_mlir"
    ),
    description="TT PyTorch FrontEnd",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    cmake_args=cmake_args,
    cmdclass={
        "install_lib": install_metal_libs,
    },
    zip_safe=False,
    install_requires=[
        "torch@https://download.pytorch.org/whl/cpu-cxx11-abi/torch-2.7.0%2Bcpu.cxx11.abi-cp310-cp310-linux_x86_64.whl",
        "numpy",
    ],
)
