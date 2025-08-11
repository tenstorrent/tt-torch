# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import gc
from tests.utils import skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from tests.runner.test_utils import (
    ModelStatus,
    import_model_loader_and_variant,
    DynamicTester,
    setup_test_discovery,
    create_test_id_generator,
)
from tests.runner.requirements import RequirementsManager

# Setup test discovery using utility functions
TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", ".."))
MODELS_ROOT, test_entries = setup_test_discovery(PROJECT_ROOT)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "test_entry",
    test_entries,
    ids=create_test_id_generator(MODELS_ROOT),
)
def test_all_models(
    test_entry, mode, op_by_op, record_property, test_metadata, request
):
    loader_path = test_entry["path"]
    variant_info = test_entry["variant_info"]

    # Ensure per-model requirements are installed, and roll back after the test
    with RequirementsManager.for_loader(loader_path):
        # FIXME - Consider cleaning this up, avoid call to import_model_loader_and_variant.
        if variant_info:
            # Unpack the tuple we stored earlier
            variant, ModelLoader, ModelVariant = variant_info
        else:
            # For models without variants
            ModelLoader, _ = import_model_loader_and_variant(loader_path, MODELS_ROOT)
            variant = None

        # Get the full test node ID
        test_node_id = request.node.nodeid
        print(f"KCM Test node ID: {test_node_id}", flush=True)

        # KCM - Testing logic to simulate crash testing.
        simulate_crash = os.environ.get("SIMULATE_CRASH", False)
        if simulate_crash:

            # CRASH SIMULATION: Crash based on node ID patterns
            crash_patterns = [
                "tests/runner/test_models.py::test_all_models[albert/masked_lm/pytorch-large_v2-full-eval]",
            ]

            for pattern in crash_patterns:
                if pattern in test_node_id:
                    import signal

                    print(
                        f"SIMULATING CRASH for: {test_node_id} w/ signal: {signal.SIGTERM}"
                    )
                    os.kill(os.getpid(), signal.SIGTERM)  # Simulate process termination

        # pytest.skip("KCM - Skipping test")

        cc = CompilerConfig()
        cc.enable_consteval = True
        cc.consteval_parameters = True
        if op_by_op:
            cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

        # Use the variant from the test_entry parameter
        loader = ModelLoader(variant=variant)

        # Get model name from the ModelLoader's ModelInfo
        model_info = ModelLoader.get_model_info(variant=variant)
        print(f"model_name: {model_info.name} status: {test_metadata.status}")

        if test_metadata.status == ModelStatus.NOT_SUPPORTED_SKIP:
            skip_full_eval_test(
                record_property,
                cc,
                model_info.name,
                bringup_status=test_metadata.skip_bringup_status,
                reason=test_metadata.skip_reason,
                model_group=model_info.group,
                forge_models_test=True,
            )

        tester = DynamicTester(
            model_info.name,
            mode,
            loader=loader,
            model_info=model_info,
            compiler_config=cc,
            record_property_handle=record_property,
            forge_models_test=True,
            **test_metadata.to_tester_args(),
        )

        results = tester.test_model()
        tester.finalize()

    # Cleanup memory after each test to prevent memory leaks
    gc.collect()
