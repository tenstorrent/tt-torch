# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from skbuild import setup
import os
from skbuild.command.install_lib import install_lib
from setuptools import find_namespace_packages
import sys
import shutil

import hashlib
import zipfile

# TODO, this is quite hacky
# The install files provided to us from tt-mlir are all in a single folder. Organize them so that
# everything is where python expects it to be.
def shuffle_wheel():
    whl = "dist/tt_torch-0.1-cp311-cp311-linux_x86_64.whl"
    print(f"Reading in {whl}")
    with zipfile.ZipFile(whl, "r") as zip_ref:
        zip_ref.extractall("wheel_contents")

    print("Moving tt-metal")
    metal_dir = os.path.join(
        os.getcwd(), "wheel_contents/tt_torch-0.1.data/data/tt-metal"
    )
    shutil.copy2(metal_dir, "wheel_contents/")
    shutil.rmtree(metal_dir)

    record_path = "wheel_contents/tt_torch-0.1.dist-info/RECORD"
    print("Populating RECORD.")
    with open(record_path, "w") as record_file:
        for root, _, files in os.walk("wheel_contents"):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, "wheel_contents")

                # Skip RECORD itself (it should be left without a hash)
                if rel_path.endswith("RECORD"):
                    record_file.write(f"{rel_path},,\n")
                    continue

                # Compute the hash
                hasher = hashlib.sha256()
                with open(file_path, "rb") as f:
                    hasher.update(f.read())
                hash_b64 = hasher.digest().hex()

                # Get file size
                file_size = os.path.getsize(file_path)

                # Write to RECORD
                record_file.write(f"{rel_path},sha256={hash_b64},{file_size}\n")

    print("Creating wheel.")
    with zipfile.ZipFile(whl, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk("wheel_contents"):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, "wheel_contents")
                zipf.write(file_path, rel_path)

    shutil.rmtree("wheel_contents", ignore_errors=True)


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

shuffle_wheel()
