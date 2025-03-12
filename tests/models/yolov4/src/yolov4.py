# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov4/reference

from tests.models.yolov4.src.downsample1 import DownSample1
from tests.models.yolov4.src.downsample2 import DownSample2
from tests.models.yolov4.src.downsample3 import DownSample3
from tests.models.yolov4.src.downsample4 import DownSample4
from tests.models.yolov4.src.downsample5 import DownSample5
from tests.models.yolov4.src.neck import Neck
from tests.models.yolov4.src.head import Head

import torch
import torch.nn as nn


class Yolov4(nn.Module):
    def __init__(self):
        super(Yolov4, self).__init__()
        self.downsample1 = DownSample1()
        self.downsample2 = DownSample2()
        self.downsample3 = DownSample3()
        self.downsample4 = DownSample4()
        self.downsample5 = DownSample5()
        self.neck = Neck()
        self.head = Head()

    def forward(self, input: torch.Tensor):
        d1 = self.downsample1(input)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)
        d5 = self.downsample5(d4)
        x20, x13, x6 = self.neck(d5, d4, d3)
        x4, x5, x6 = self.head(x20, x13, x6)

        return x4, x5, x6

    @staticmethod
    def from_random_weights():
        model = Yolov4()
        model.eval()

        new_state_dict = {}
        for name, parameter in model.state_dict().items():
            if isinstance(parameter, torch.FloatTensor):
                new_state_dict[name] = parameter

        model.load_state_dict(new_state_dict)
        return model
