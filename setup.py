# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from skbuild import setup
import os
from skbuild.command.install_lib import install_lib
import glob
import sys
import shutil


from setuptools.command.build_py import build_py as _build_py
from setuptools import find_namespace_packages
import os


class build_py_with_torch_mlir(_build_py):
    def finalize_options(self):
        super().finalize_options()

        torch_mlir_path = os.path.join(
            "third_party",
            "torch-mlir",
            "src",
            "torch-mlir-build",
            "python_packages",
            "torch_mlir",
        )

        assert os.path.exists(torch_mlir_path)
        extra_packages = find_namespace_packages(
            where=torch_mlir_path, include=["torch_mlir*"]
        )
        self.packages.extend(extra_packages)
        self.package_dir.update(
            {
                pkg: os.path.join(torch_mlir_path, *pkg.split("."))
                for pkg in extra_packages
            }
        )

    def find_data_files(self, package, src_dir):
        data_files = super().find_data_files(package, src_dir)

        # Add all non-Python files from torch_mlir packages
        if package.startswith("torch_mlir"):
            files = [
                f
                for f in os.listdir(src_dir)
                if os.path.isfile(os.path.join(src_dir, f))
            ]
            for file in files:
                if not file.endswith(".py") and not file.endswith(".pyc"):
                    full_path = os.path.join(src_dir, file)
                    # Check if file exists before adding
                    if os.path.isfile(full_path):
                        data_files.append(full_path)

        return data_files


class install_metal_libs(install_lib):
    def run(self):
        install_lib.run(self)
        install_path = os.path.join(self.install_dir, "tt_mlir")
        os.makedirs(install_path, exist_ok=True)
        ttmlir_opt = os.path.abspath(
            os.path.join(
                os.getcwd(),
                "third_party",
                "tt-xla",
                "src",
                "tt-xla",
                "third_party",
                "tt-mlir",
                "build",
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
                "tt-xla",
                "src",
                "tt-xla",
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
        dest_tools_dir = os.path.join(self.install_dir, "tt-metal", "tools")
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
        # copy everything from skbuild cmake-install/tt-metal to self.install_dir/tt-metal
        src_metal_dir = "_skbuild/linux-x86_64-3.11/cmake-install/tt-metal"
        dest_metal_dir = os.path.join(self.install_dir, "tt-metal")
        if os.path.exists(src_metal_dir):
            os.makedirs(dest_metal_dir, exist_ok=True)
            shutil.copytree(src_metal_dir, dest_metal_dir, dirs_exist_ok=True)

        # Copy shared libraries from skbuild location to tt_torch
        lib_dest_dir = os.path.join(self.install_dir)
        os.makedirs(lib_dest_dir, exist_ok=True)

        # Find the skbuild cmake-install directory - include versioned libraries
        skbuild_lib_pattern = "_skbuild/*/cmake-install/lib/*.so*"
        so_files = glob.glob(skbuild_lib_pattern)

        if not so_files:
            assert False
        else:
            print(f"Found {len(so_files)} shared libraries to copy:")
            for so_file in so_files:
                print(f"  Copying {so_file} to {lib_dest_dir}")
                shutil.copy2(so_file, lib_dest_dir)


# Compile time env vars
os.environ["DONT_OVERRIDE_INSTALL_PATH"] = "1"

install_requires = [
    "torch@https://download.pytorch.org/whl/cpu/torch-2.7.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl",  # pytorch does not always resolve as CPU pkg by default
    "torch-xla@https://pypi.eng.aws.tenstorrent.com/torch-xla/torch_xla-2.9.0%2Bgit1adbe97-cp311-cp311-linux_x86_64.whl",
    "stablehlo@https://github.com/openxla/stablehlo/releases/download/v1.0.0/stablehlo-1.0.0.1715728102%2B6051bcdf-cp311-cp311-linux_x86_64.whl",
    "torchvision@https://download.pytorch.org/whl/cpu/torchvision-0.22.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl",
    "numpy",
    "onnx==1.17.0",
    "onnxruntime",
    "ml_dtypes",
]

cmake_args = [
    "-GNinja",
    "-DBUILD_TTRT=OFF",
]
if "--code_coverage" in sys.argv:
    cmake_args += [
        "-DCODE_COVERAGE=ON",
    ]
    sys.argv.remove("--code_coverage")

build_perf = "--build_perf" in sys.argv
if build_perf:
    cmake_args += [
        "-DTT_RUNTIME_ENABLE_PERF_TRACE=ON",
    ]

    # Additional python dependencies are required for profiling
    # and perf analysis tools provided by metal process_ops
    install_requires.extend(["pyyaml", "click", "loguru", "pandas", "seaborn"])
    sys.argv.remove("--build_perf")

if "--build_runtime_debug" in sys.argv:
    cmake_args += [
        "-DTT_RUNTIME_DEBUG=ON",
    ]
    sys.argv.remove("--build_runtime_debug")

if "--build_op_model" in sys.argv:
    cmake_args += ["-DTTMLIR_ENABLE_OPMODEL=ON"]
    sys.argv.remove("--build_op_model")


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
    url="https://github.com/tenstorrent/tt-torch",
    packages=find_namespace_packages(include=["tt_torch*"]),
    description="TT PyTorch FrontEnd",
    long_description=long_description,
    long_description_content_type="text/markdown",
    cmake_args=cmake_args,
    cmdclass={
        "build_py": build_py_with_torch_mlir,
        "install_lib": install_metal_libs,
    },
    zip_safe=False,
    install_requires=install_requires,
    include_package_data=True,
    entry_points={
        "torch_dynamo_backends": [
            "tt = tt_torch.dynamo.backend:backend",
        ],
        "console_scripts": [
            "tt_profile = tt_torch.tools.tt_profile:main",
        ],
    },
)
