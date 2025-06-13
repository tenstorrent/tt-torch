# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.utils import skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig

# List all priority red group models here that are not started or WIP for reporting purposes
# Model names are best guesses, may change when tests are worked on, added.
red_models = [
    # PYTORCH Models ======================================================
    # Phi Models - https://github.com/tenstorrent/tt-torch/issues/330
    "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3.5-MoE-instruct",
    "microsoft/Phi-3.5-vision-instruct",
    "microsoft/phi-4",
    # Surya-OCR Models - https://github.com/tenstorrent/tt-torch/issues/425
    "surya-ocr-detection",
    "surya-ocr-recognition",
    "surya-ocr-layout",
    "surya-ocr-table_recognition",
    # ONNX Models =========================================================
    "BEVDepth",  # https://github.com/tenstorrent/tt-torch/issues/348
    # swin transformer v2 - https://github.com/tenstorrent/tt-torch/issues/333
]

# Generate XML report for red priority models
def test_wip_priority_red_models(record_property):
    for model_name in red_models:
        print(f"Generating report for model_name: {model_name}", flush=True)
        skip_full_eval_test(
            record_property,
            CompilerConfig(),
            model_name,
            bringup_status="NOT_STARTED",
            reason="Model is not started or is WIP",
            model_group="red",
            skip=False,
        )
