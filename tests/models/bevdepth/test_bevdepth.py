# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/Megvii-BaseDetection/BEVDepth

import torch
import numpy as np
import pytest

from tests.models.bevdepth.src.base_bev_depth import BaseBEVDepth
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        backbone_conf = {
            "x_bound": [-51.2, 51.2, 0.8],
            "y_bound": [-51.2, 51.2, 0.8],
            "z_bound": [-5, 3, 8],
            "input_size": (256, 704),
            "d_bound": [2.0, 58.0, 0.5],
            "final_dim": (128, 128),
            "output_channels": 80,
            "downsample_factor": 16,
            "img_backbone_conf": {
                "num_layers": [3, 4, 6, 3],
                "layer_strides": [1, 2, 2, 2],
                "num_filters": [64, 128, 256, 512],
                "layer_dims": [64, 128, 256, 512],
                "pretrained": None,
            },
            "img_neck_conf": {
                "in_channels": [64, 128, 256, 512],
                "out_channels": 256,
                "start_level": 0,
                "num_outs": 4,
            },
            "depth_net_conf": {"in_channels": 256, "mid_channels": 512},
        }

        head_conf = {
            "in_channels": 80,
            "tasks": [
                dict(num_class=1, class_names=["car"]),
                dict(num_class=2, class_names=["truck", "construction_vehicle"]),
                dict(num_class=2, class_names=["bus", "trailer"]),
                dict(num_class=1, class_names=["barrier"]),
                dict(num_class=2, class_names=["motorcycle", "bicycle"]),
                dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
            ],
            "common_heads": {
                "reg": (2, 2),
                "height": (1, 2),
                "dim": (3, 2),
                "rot": (2, 2),
                "vel": (2, 2),
            },
            "bbox_coder": {
                "type": "CenterPointBBoxCoder",
                "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                "max_num": 500,
                "score_threshold": 0.1,
                "out_size_factor": 4,
                "voxel_size": [0.2, 0.2, 8],
                "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                "code_size": 9,
            },
        }

        model = BaseBEVDepth(backbone_conf=backbone_conf, head_conf=head_conf)
        return model.to(torch.float16)

    def _load_inputs(self):
        # Create dummy input data
        batch_size = 1
        num_sweeps = 1
        num_cameras = 6
        img_height = 256
        img_width = 704

        # Image tensor
        x = torch.randn(
            batch_size * num_sweeps * num_cameras, 3, img_height, img_width
        ).to(torch.float16)

        # Camera matrices
        mats_dict = {
            "sensor2ego_mats": torch.eye(4)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, num_sweeps, num_cameras, 1, 1)
            .to(torch.float16),
            "intrin_mats": torch.eye(4)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, num_sweeps, num_cameras, 1, 1)
            .to(torch.float16),
            "ida_mats": torch.eye(4)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, num_sweeps, num_cameras, 1, 1)
            .to(torch.float16),
            "sensor2sensor_mats": torch.eye(4)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, num_sweeps, num_cameras, 1, 1)
            .to(torch.float16),
            "bda_mat": torch.eye(4)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .to(torch.float16),
        }

        return (x, mats_dict)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    # [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    # ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
    [None],
    ids=["full"],
)
def test_bevdepth(record_property, mode, op_by_op):
    model_name = "BEVDepth"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name,
        mode,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group="red",
    )
    with torch.no_grad():
        tester.test_model()
    tester.finalize()
