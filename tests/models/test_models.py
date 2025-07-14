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


def get_models_root(project_root: str) -> str:
    """Return the filesystem path to the given module, supporting both installed and source-tree use cases."""
    module_name = "third_party.tt_forge_models"
    spec = importlib.util.find_spec(module_name)
    if spec:
        if spec.submodule_search_locations:
            return spec.submodule_search_locations[0]
        elif spec.origin:
            return os.path.dirname(os.path.abspath(spec.origin))

    # Derive filesystem path from module name
    rel_path = os.path.join(*module_name.split("."))
    fallback = os.path.join(project_root, rel_path)
    print(f"No installed {module_name}; falling back to {fallback}")
    return fallback


# breakpoint()
TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", ".."))
MODELS_ROOT = get_models_root(PROJECT_ROOT)

# Add the models root to sys.path so relative imports work
if MODELS_ROOT not in sys.path:
    sys.path.insert(0, MODELS_ROOT)

loader_paths = {}
for root, dirs, files in os.walk(MODELS_ROOT):
    if os.path.basename(root) == "pytorch" and "loader.py" in files:
        loader_paths[os.path.join(root, "loader.py")] = []


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


def get_model_variants(loader_path):
    try:
        loader = import_model_loader(loader_path)
        variants = loader.query_available_variants()
        for variant in variants.keys():
            loader_paths[loader_path].append(variant)

    except:
        print(f"Cannor import path: {loader_path}")


for path in loader_paths.keys():
    get_model_variants(path)

# Create test entries combining loader paths and variants
test_entries = []
for loader_path, variants in loader_paths.items():
    if variants:  # Model has variants
        for variant in variants:
            test_entries.append({"path": loader_path, "variant": variant})
    else:  # Model has no variants
        test_entries.append({"path": loader_path, "variant": None})


def generate_test_id(test_entry):
    """Generate test ID from test entry."""
    model_path = os.path.relpath(os.path.dirname(test_entry["path"]), MODELS_ROOT)
    if test_entry["variant"]:
        return f"{model_path}-{test_entry['variant']}"
    else:
        return model_path


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
    "test_entry",
    test_entries,
    ids=generate_test_id,
)
def test_all_models(test_entry, mode, op_by_op, record_property):
    loader_path = test_entry["path"]
    variant = test_entry["variant"]

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

    # Use the variant from the test_entry parameter
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
