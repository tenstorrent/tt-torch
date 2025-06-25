# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import sys
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
import importlib.util
import torch
import inspect

# breakpoint()
TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", ".."))
MODELS_ROOT = os.path.join(PROJECT_ROOT, "third_party", "tt_forge_models")

# Add the models root to sys.path so relative imports work
if MODELS_ROOT not in sys.path:
    sys.path.insert(0, MODELS_ROOT)

loader_paths = []
for root, dirs, files in os.walk(MODELS_ROOT):
    if os.path.basename(root) == "pytorch" and "loader.py" in files:
        loader_paths.append(os.path.join(root, "loader.py"))


def import_model_loader(loader_path):
    # Import the base module first to ensure it's available
    import sys

    models_parent = os.path.dirname(MODELS_ROOT)
    if models_parent not in sys.path:
        sys.path.insert(0, models_parent)

    # Import the tt_forge_models module to make relative imports work
    import tt_forge_models

    # Get the relative path from MODELS_ROOT to construct proper module name
    rel_path = os.path.relpath(loader_path, MODELS_ROOT)
    module_path = "tt_forge_models." + rel_path.replace(os.sep, ".").replace(".py", "")

    spec = importlib.util.spec_from_file_location(module_path, loader_path)
    mod = importlib.util.module_from_spec(spec)

    # Set the module's __package__ for relative imports to work
    loader_dir = os.path.dirname(loader_path)
    package_name = "tt_forge_models." + os.path.relpath(
        loader_dir, MODELS_ROOT
    ).replace(os.sep, ".")
    mod.__package__ = package_name
    mod.__name__ = module_path

    # Add the module to sys.modules to support relative imports
    sys.modules[module_path] = mod

    spec.loader.exec_module(mod)
    return mod.ModelLoader


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [None],
    ids=["full"],
    # [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    # ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "loader_path",
    loader_paths,
    ids=lambda p: os.path.relpath(os.path.dirname(p), MODELS_ROOT),
)
def test_all_models(loader_path, mode, op_by_op, record_property):
    ModelLoader = import_model_loader(loader_path)

    class DynamicTester(ModelTester):
        def _load_model(self):
            # Check if load_model method supports dtype_override parameter
            sig = inspect.signature(self.loader.load_model)
            if "dtype_override" in sig.parameters:
                return self.loader.load_model(dtype_override=torch.bfloat16)
            else:
                return self.loader.load_model()

        def _load_inputs(self):
            # Check if load_inputs method supports dtype_override parameter
            sig = inspect.signature(self.loader.load_inputs)
            if "dtype_override" in sig.parameters:
                return self.loader.load_inputs(dtype_override=torch.bfloat16)
            else:
                return self.loader.load_inputs()

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    # TODO - Figure out how to run variants in this test.
    # they are normally pytest params after querying via
    # available_variants = ModelLoader.query_available_variants()
    variant = None
    loader = ModelLoader(variant=variant)

    # Get model name from the ModelLoader's ModelInfo
    model_info = ModelLoader.get_model_info(variant=variant)
    model_name = model_info.name

    tester = DynamicTester(
        model_name,
        mode,
        loader=loader,
        compiler_config=cc,
        assert_atol=False,
        assert_pcc=True,
        record_property_handle=record_property,
    )
    results = tester.test_model()
    tester.finalize()
