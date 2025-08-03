# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelStatus


test_config = {
    "gpt_neo/pytorch-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vovnet/pytorch-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hardnet/pytorch-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "blackhole": {
                "required_pcc": 0.97,
            },
        },
    },
    "qwen/casual_lm/pytorch-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "clip/pytorch-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolo_x/pytorch-full-eval": {
        "assert_pcc": False,
    },
    "wide_resnet/pytorch-full-eval": {
        "required_pcc": 0.96,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bloom/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xglm/pytorch-xglm-564M-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xglm/pytorch-xglm-1.7B-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnet/pytorch-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-790m-hf-full-eval": {
        "required_pcc": 0.95,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "openpose/v2/pytorch-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-albert-xxlarge-v2-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-albert-large-v2-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov3/pytorch-base-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov4/pytorch-base-full-eval": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "flan_t5/pytorch-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "musicgen_small/pytorch-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "falcon/pytorch-full-eval": {
        "assert_pcc": False,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "yolov5/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-albert-base-v2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "albert/masked_lm/pytorch-albert-xlarge-v2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "alexnet/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "autoencoder_linear/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bart/pytorch-large-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bert/pytorch-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "bert/pytorch-large-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "codegen/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-base_distilled-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-small-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "deit/pytorch-tiny-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet121-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet161-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet169-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "densenet/pytorch-densenet201-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "distilbert/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla102-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla102x2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla102x-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla169-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla34-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla46_c-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla46x_c-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60x_c-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dla/pytorch-dla60x-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dpr/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "dpr/reader/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b0-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b3-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b4-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b5-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b6-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "efficientnet/pytorch-efficientnet_b7-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "ghostnet/pytorch-ghostnet_100-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "ghostnet/pytorch-ghostnet_100.in1k-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18.ms_aug_in1k-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w18_small_v2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w30-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w32-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w40-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w44-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w48-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "hrnet/pytorch-hrnet_w64-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-1.4b-hf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "arch_overrides": {
            "blackhole": {
                "status": ModelStatus.NOT_SUPPORTED_SKIP,
                "skip_reason": "Takes forever on blackhole runner",
                "skip_bringup_status": "FAILED_RUNTIME",
            },
        },
    },
    "mamba/pytorch-mamba-370m-hf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mgp_str_base/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b16_224_miil-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b32_224-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_l32_224-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_s16_224-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_s32_224-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mlp_mixer/pytorch-mixer_b16_224_miil_in21k-full-eval": {
        "required_pcc": 0.97,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mnist/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv1/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mobilenetv2/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "nanogpt/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1_5/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_040-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_064-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_080-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_120-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_160-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "regnet/pytorch-regnet_y_320-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext101_32x8d-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext101_32x8d_wsl-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "resnext/pytorch-resnext50_32x4d-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "roberta/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b0-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b1-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b3-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b4-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "segformer/pytorch-mit_b5-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "squeezebert/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "swin/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "unet/pytorch-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg11_bn-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg11-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg13_bn-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg13-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg16_bn-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg16-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg19_bn-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vgg/pytorch-vgg19-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "vit/pytorch-large-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception41-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception65-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception71-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "xception/pytorch-xception71.tf_in1k-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "roberta/masked_lm/pytorch-xlm_base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "llama/causal_lm/pytorch-3b-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "unet/torch_hub/pytorch-brain_segmentation-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mamba/pytorch-mamba-2.8b-hf-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
        "arch_overrides": {
            "blackhole": {
                "status": ModelStatus.NOT_SUPPORTED_SKIP,
                "skip_reason": "Takes forever on blackhole runner",
                "skip_bringup_status": "FAILED_RUNTIME",
            },
        },
    },
    "deit/pytorch-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "required_pcc": 0.98,
    },
    "mlp_mixer/lucidrains/pytorch-base-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "mistral/pytorch-ministral_3b_instruct-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_b16_224-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_b16_224_in21k-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_l16_224-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_l16_224_in21k-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "mlp_mixer/pytorch-mixer_b16_224.goog_in21k-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "phi2/causal_lm/pytorch-microsoft/phi-2-full-eval": {
        "status": ModelStatus.KNOWN_FAILURE_XFAIL, # High memory killed
        "assert_pcc": False,
    },
    "phi2/causal_lm/pytorch-microsoft/phi-2-pytdml-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
        "assert_pcc": False,
    },
    "phi2/token_classification/pytorch-microsoft/phi-2-full-eval": {
        "required_pcc": 0.97,  # PCC is ND https://github.com/tenstorrent/tt-torch/issues/1129
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi2/token_classification/pytorch-microsoft/phi-2-pytdml-full-eval": {
        "required_pcc": 0.97,  # PCC is ND https://github.com/tenstorrent/tt-torch/issues/1129
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi2/sequence_classification/pytorch-microsoft/phi-2-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi2/sequence_classification/pytorch-microsoft/phi-2-pytdml-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1_5/token_classification/pytorch-microsoft/phi-1_5-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1_5/causal_lm/pytorch-microsoft/phi-1_5-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
    "phi1_5/sequence_classification/pytorch-microsoft/phi-1_5-full-eval": {
        "status": ModelStatus.EXPECTED_PASSING,
    },
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
