# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import sys
from tests.utils import skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from tests.runner.test_utils import (
    ModelStatus,
    get_models_root,
    import_model_loader_and_variant,
    get_model_variants,
    generate_test_id,
    get_tester_args,
    DynamicTester,
)


@pytest.fixture(autouse=True)
def log_test_name(request):
    print(f"\nRunning {request.node.nodeid}", flush=True)


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

for path in loader_paths.keys():
    get_model_variants(path, loader_paths, MODELS_ROOT)

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


def _generate_test_id(test_entry):
    """Generate test ID from test entry using the utility function."""
    return generate_test_id(test_entry, MODELS_ROOT)


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
    ids=_generate_test_id,
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
        ModelLoader, _ = import_model_loader_and_variant(loader_path, MODELS_ROOT)
        variant = None

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

    print(f"model_name: {model_info.name} status: {test_metadata.status}")

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
