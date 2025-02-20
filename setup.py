# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from skbuild import setup
import os
import glob
from skbuild.command.install_lib import install_lib
import shutil
import sys


class install_metal_libs(install_lib):
    def run(self):
        install_lib.run(self)
        install_path = os.path.join(self.install_dir, "tt_mlir")
        # Copy third_party/tt-mlir/src/tt-mlir-build/bin/ttmlir-opt into lib
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
setup(
    name="tt_torch",
    version="0.1",
    author="Aleks Knezevic",
    author_email="aknezevic@tenstorrent.com",
    license="Apache-2.0",
    homepage="https://github.com/tenstorrent/tt-torch",
    packages=[
        "tt_torch",
    ],
    description="TT PyTorch FrontEnd",
    long_description="",
    cmake_args=cmake_args,
    cmdclass={
        "install_lib": install_metal_libs,
    },
    zip_safe=False,
)
