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
        lib_dir = os.path.abspath(
            os.path.join(
                os.getcwd(), "_skbuild", "linux-x86_64-3.11", "cmake-install", "lib"
            )
        )
        if os.path.exists(lib_dir):
            for file in glob.glob(os.path.join(lib_dir, "*")):
                if os.path.isfile(file):
                    install_path = os.path.join(self.install_dir, "tt_torch")
                    os.makedirs(install_path, exist_ok=True)
                    self.copy_file(file, install_path)

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
        os.makedirs(os.path.join(self.install_dir, "tt_torch"), exist_ok=True)
        self.copy_file(ttmlir_opt, os.path.join(self.install_dir, "tt_torch"))

        metal_dir = os.path.abspath(
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
            )
        )
        os.makedirs(self.install_dir, exist_ok=True)

        tt_metal_folders_to_ignore = [
            ".cpmcache",
            ".git",
        ]

        def ignore(folder, contents):
            relative_folder = os.path.relpath(folder, start=metal_dir)

            ignored_items = [
                item
                for item in contents
                if any(
                    os.path.join(relative_folder, item).startswith(ignore)
                    for ignore in tt_metal_folders_to_ignore
                )
            ]

            return ignored_items

        install_path = os.path.join(self.install_dir, "tt_torch" "tt_metal")
        os.makedirs(install_path, exist_ok=True)
        shutil.copytree(metal_dir, install_path, dirs_exist_ok=True, ignore=ignore)


# Compile time env vars
os.environ["DONT_OVERRIDE_INSTALL_PATH"] = "1"

cmake_args = [
    "-GNinja",
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
