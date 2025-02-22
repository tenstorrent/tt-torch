# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from skbuild import setup
import os
import glob
from skbuild.command.install_lib import install_lib
from setuptools import find_namespace_packages
import sys


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


# Compile time env vars
os.environ["WHEEL_BUILD_CUSTOM_INSTALL_PATH"] = "1"

cmake_args = [
    "-GNinja",
    "-DBUILD_TTRT=OFF",
]
if "--code_coverage" in sys.argv:
    cmake_args += [
        "-DCODE_COVERAGE=ON",
    ]
    sys.argv.remove("--code_coverage")

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="tt_torch",
    version="0.1",
    author="Aleks Knezevic",
    author_email="aknezevic@tenstorrent.com",
    license="Apache-2.0",
    homepage="https://github.com/tenstorrent/tt-torch",
    packages=find_namespace_packages(include=["tt_torch.*"]),
    description="TT PyTorch FrontEnd",
    long_description=long_description,
    long_description_content_type="text/markdown",
    cmake_args=cmake_args,
    cmdclass={
        "install_lib": install_metal_libs,
    },
    zip_safe=False,
    install_requires=[
        "torch@https://download.pytorch.org/whl/cpu-cxx11-abi/torch-2.5.0%2Bcpu.cxx11.abi-cp311-cp311-linux_x86_64.whl",
        "numpy",
    ],
)
