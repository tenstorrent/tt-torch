# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import sys
from tests.utils import ModelTester, skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
import importlib.util
import torch
import inspect
from tests.runner.test_config import ModelStatus


@pytest.fixture(autouse=True)
def log_test_name(request):
    print(f"\nRunning {request.node.nodeid}", flush=True)


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


def import_model_loader_and_variant(loader_path):
    # Import the base module first to ensure it's available
    import sys

    models_parent = os.path.dirname(MODELS_ROOT)
    if models_parent not in sys.path:
        sys.path.insert(0, models_parent)

    # Import the tt_forge_models module to make relative imports work
    # import tt_forge_models

    # Get the relative path from MODELS_ROOT to construct proper module name
    rel_path = os.path.relpath(loader_path, MODELS_ROOT)
    rel_path_without_ext = rel_path.replace(".py", "")

    # Use different/dummy module name to avoid conflicts with real package name
    module_path = "tt-forge-models." + rel_path_without_ext.replace(os.sep, ".")

    spec = importlib.util.spec_from_file_location(module_path, location=loader_path)
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

    # Find ModelVariant class in the module
    ModelVariant = None
    for name, obj in mod.__dict__.items():
        if name == "ModelVariant":
            ModelVariant = obj
            break

    return mod.ModelLoader, ModelVariant


def get_model_variants(loader_path):
    try:
        # Import both the ModelLoader and ModelVariant class from the same module
        ModelLoader, ModelVariant = import_model_loader_and_variant(loader_path)
        variants = ModelLoader.query_available_variants()
        for variant in variants.keys():
            # Store the variant, ModelLoader class, and ModelVariant class together
            loader_paths[loader_path].append((variant, ModelLoader, ModelVariant))

    except Exception as e:
        print(f"Cannot import path: {loader_path}: {e}")


for path in loader_paths.keys():
    get_model_variants(path)

# Create test entries combining loader paths and variants
test_entries = []

# Store variant info along with the ModelLoader and ModelVariant classes
for loader_path, variant_tuples in loader_paths.items():
    if variant_tuples:  # Model has variants
        for variant_tuple in variant_tuples:
            # Each tuple contains (variant, ModelLoader, ModelVariant)
            test_entries.append({"path": loader_path, "variant_info": variant_tuple})
    else:  # Model has no variants
        test_entries.append({"path": loader_path, "variant_info": None})


def generate_test_id(test_entry):
    """Generate test ID from test entry."""
    model_path = os.path.relpath(os.path.dirname(test_entry["path"]), MODELS_ROOT)
    variant_info = test_entry["variant_info"]

    if variant_info:
        variant, _, _ = variant_info  # Unpack the tuple to get just the variant
        return f"{model_path}-{variant}"
    else:
        return model_path


def get_tester_args(test_metadata):

    # FIXME - Do we even need to go throught these, why not more directly?
    args = {}

    if test_metadata.assert_pcc is not None:
        args["assert_pcc"] = test_metadata.assert_pcc

    if test_metadata.assert_atol is not None:
        args["assert_atol"] = test_metadata.assert_atol

    if test_metadata.pcc is not None:
        args["required_pcc"] = test_metadata.pcc

    if test_metadata.relative_atol is not None:
        args["relative_atol"] = test_metadata.relative_atol

    return args


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
def test_all_models(test_entry, mode, op_by_op, record_property, test_metadata):
    loader_path = test_entry["path"]
    variant_info = test_entry["variant_info"]

    # FIXME - Consider cleaning this up, avoid call to import_model_loader_and_variant.
    if variant_info:
        # Unpack the tuple we stored earlier
        variant, ModelLoader, ModelVariant = variant_info
    else:
        # For models without variants
        ModelLoader, _ = import_model_loader_and_variant(loader_path)
        variant = None

    # FIXME - Consider moving this.
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

    print(f"Running {model_info.name} status: {test_metadata.status}")

    if test_metadata.status == ModelStatus.NOT_SUPPORTED_SKIP:
        # FIXME - Add skip_msg and bringup_status to test_config.
        skip_test_msg = "blah blah"
        if skip_test_msg:
            skip_full_eval_test(
                record_property,
                cc,
                model_info.name,
                bringup_status="FAILED_RUNTIME",
                reason=skip_test_msg,
                model_group=model_info.group,
            )

    # Extract args from test config file if present.
    args = get_tester_args(test_metadata)

    tester = DynamicTester(
        model_info.name,
        mode,
        loader=loader,
        compiler_config=cc,
        record_property_handle=record_property,
        **args,
    )
    # results = tester.test_model()
    # tester.finalize()
