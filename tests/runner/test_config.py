# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelStatus


# TODO - Add a step that verifies configs are valid.
test_config = {
    "gpt_neo/pytorch-full-eval": {
        "required_pcc": 0.98,
    },
    "vovnet/pytorch-full-eval": {
        "assert_pcc": False,
    },
    "hardnet/pytorch-full-eval": {
        "required_pcc": 0.98,
    },
    "qwen/casual_lm/pytorch-full-eval": {
        "assert_pcc": False,
    },
    "clip/pytorch-full-eval": {
        "assert_pcc": False,
    },
    "yolo_x/pytorch-full-eval": {
        "assert_pcc": False,
    },
    "wide_resnet/pytorch-full-eval": {
        "required_pcc": 0.96,
    },
    "efficientnet/pytorch-full-eval": {
        "assert_pcc": False,
    },
    "bloom/pytorch-full-eval": {
        "assert_pcc": False,
    },
    "xglm/pytorch-xglm-564M-full-eval": {
        "assert_pcc": False,
    },
    "xglm/pytorch-xglm-1.7B-full-eval": {
        "assert_pcc": False,
    },
    "resnet/pytorch-full-eval": {
        "required_pcc": 0.97,
    },
    "mamba/pytorch-mamba-790m-hf-full-eval": {
        "required_pcc": 0.95,
    },
    "openpose/v2/pytorch-full-eval": {
        "assert_pcc": False,
    },
    "albert/masked_lm/pytorch-albert-xxlarge-v2-full-eval": {
        "required_pcc": 0.97,
    },
    "albert/masked_lm/pytorch-albert-large-v2-full-eval": {
        "required_pcc": 0.97,
    },
    "deit/pytorch-full-eval": {
        "required_pcc": 0.97,
        "relative_atol": 0.015,
    },
    "yolov3/pytorch-base-full-eval": {
        "required_pcc": 0.97,
    },
    "yolov4/pytorch-base-full-eval": {
        "required_pcc": 0.98,
    },
    "flan_t5/pytorch-full-eval": {
        "assert_pcc": False,
    },
    "musicgen_small/pytorch-full-eval": {
        "assert_pcc": False,
    },
    "falcon/pytorch-full-eval": {
        "assert_pcc": False,
    },
    # Skip models converted from skip_models section
    "yolov10/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "TorchMlirCompilerError: Lowering Torch Backend IR -> StableHLO Backend IR failed",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "qwen/token_classification/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Out of Memory: Not enough space to allocate 135790592 B DRAM buffer across 12 banks, where each bank needs to store 11317248 B",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "t5/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "vgg19_unet/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Out of Memory: Not enough space to allocate 84213760 B L1 buffer across 64 banks, where each bank needs to store 1315840 B - https://github.com/tenstorrent/tt-torch/issues/729",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "yolov9/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "RuntimeError: TT_FATAL @ Inputs must be of bfloat16 or bfloat8_b type",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "detr/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Out of Memory: Not enough space to allocate 4294967296 B DRAM buffer across 12 banks",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "monodepth2/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "tt-forge-models needs to be updated to use get_file() api for loading .pth files",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "glpn_kitti/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "oft/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Out of Memory: Not enough space to allocate 2902982656 B DRAM buffer across 12 banks - https://github.com/tenstorrent/tt-torch/issues/727",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "yolov8/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "RuntimeError: TT_FATAL @ Inputs must be of bfloat16 or bfloat8_b type",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "vilt/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "RuntimeError: cannot sample n_sample <= 0 samples",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "whisper/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": 'ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")',
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "yolo_v6/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Needs fix import library in tt-forge-models",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "gliner_model/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "AttributeError: 'function' object has no attribute 'parameters'",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "deepseek/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Fix KILLED",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "deepseek/qwen/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Fix KILLED",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "mistral/pixtral/pytorch-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "Fix KILLED",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "bi_rnn_crf/pytorch-lstm-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "need 'aten::sort' torch-mlir -> stablehlo + mlir support: failed to legalize operation 'torch.constant.bool' - https://github.com/tenstorrent/tt-torch/issues/724",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
    "bi_rnn_crf/pytorch-gru-full-eval": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "skip_reason": "need 'aten::sort' torch-mlir -> stablehlo + mlir support: failed to legalize operation 'torch.constant.bool' - https://github.com/tenstorrent/tt-torch/issues/724",
        "skip_bringup_status": "FAILED_RUNTIME",
    },
}
