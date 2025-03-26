# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


class BEVDepthHead(nn.Module):
    def __init__(self, in_channels, tasks, common_heads, bbox_coder=None):
        super(BEVDepthHead, self).__init__()

        self.tasks = tasks
        self.in_channels = in_channels

        # Build classification and regression heads
        self.heads = nn.ModuleList()
        for task in tasks:
            heads = {}
            for head_name, head_dim in common_heads.items():
                out_channels, num_conv = head_dim
                conv_layers = []
                c_in = in_channels
                for i in range(num_conv - 1):
                    conv_layers.append(nn.Conv2d(c_in, c_in, 3, padding=1, bias=True))
                    conv_layers.append(nn.ReLU(True))
                conv_layers.append(
                    nn.Conv2d(c_in, out_channels, 3, padding=1, bias=True)
                )
                heads[head_name] = nn.Sequential(*conv_layers)

            # Classification head
            heads["cls"] = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(in_channels, task["num_class"], 3, padding=1),
            )

            self.heads.append(nn.ModuleDict(heads))

    def forward(self, x):
        preds = []
        for task_id, task_head in enumerate(self.heads):
            pred = {}
            # Classification branch
            pred["cls"] = clip_sigmoid(task_head["cls"](x))

            # Regression branches
            for key, head in task_head.items():
                if key != "cls":
                    pred[key] = head(x)

            preds.append(pred)

        return preds


__all__ = ["BEVDepthHead"]

bev_backbone_conf = dict(
    type="ResNet",
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck_conf = dict(
    type="SECONDFPN",
    in_channels=[160, 320, 640],
    upsample_strides=[2, 4, 8],
    out_channels=[64, 64, 128],
)


def size_aware_circle_nms(dets, thresh_scale, post_max_size=83):
    """Circular NMS.
    An object is only counted as positive if no other center
    with a higher confidence exists within a radius r using a
    bird-eye view distance metric.
    Args:
        dets (torch.Tensor): Detection results with the shape of [N, 3].
        thresh (float): Value of threshold.
        post_max_size (int): Max number of prediction to be kept. Defaults
            to 83
    Returns:
        torch.Tensor: Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    dx1 = dets[:, 2]
    dy1 = dets[:, 3]
    yaws = dets[:, 4]
    scores = dets[:, -1]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist_x = abs(x1[i] - x1[j])
            dist_y = abs(y1[i] - y1[j])
            dist_x_th = (
                abs(dx1[i] * np.cos(yaws[i]))
                + abs(dx1[j] * np.cos(yaws[j]))
                + abs(dy1[i] * np.sin(yaws[i]))
                + abs(dy1[j] * np.sin(yaws[j]))
            )
            dist_y_th = (
                abs(dx1[i] * np.sin(yaws[i]))
                + abs(dx1[j] * np.sin(yaws[j]))
                + abs(dy1[i] * np.cos(yaws[i]))
                + abs(dy1[j] * np.cos(yaws[j]))
            )
            # ovr = inter / areas[j]
            if (
                dist_x <= dist_x_th * thresh_scale / 2
                and dist_y <= dist_y_th * thresh_scale / 2
            ):
                suppressed[j] = 1
    return keep[:post_max_size]
