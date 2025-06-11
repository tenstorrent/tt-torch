# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import urllib
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend


class ThisTester(ModelTester):
    def _load_model(self):
        """
        Test UNet for brain MRI segmentation
        The model is from https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
        """
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=True,
        )
        model = model.to(torch.bfloat16)
        return model

    def _load_inputs(self):
        url, filename = (
            "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png",
            "TCGA_CS_4944.png",
        )
        try:
            urllib.URLopener().retrieve(url, filename)
        except:
            urllib.request.urlretrieve(url, filename)

        input_image = Image.open(filename)
        m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=m, std=s),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = torch.stack([input_tensor] * 4)
        input_batch = input_batch.to(torch.bfloat16)
        return input_batch


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_unet_brain(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "Unet-brain"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.automatic_parallelization = True
    cc.mesh_shape = [1, 2]
    cc.dump_debug = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name, mode, compiler_config=cc, record_property_handle=record_property
    )
    results = tester.test_model()
    if mode == "eval":
        output = results[0]
        for i in range(output.shape[0]):
            print(f"Sample {i+1}:")
            print(torch.round(output[i]))

    tester.finalize()
