# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from PIL import Image
import torch
import pytest
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file


class ThisTester(ModelTester):
    def _load_model(self):
        model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0", torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        torch.set_float32_matmul_precision(["high", "highest"][0])
        image_size = (1024, 1024)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return model

    def _load_inputs(self):
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.image = Image.open(str(image_file))
        inputs = self.transform_image(self.image).unsqueeze(0).to(dtype=torch.bfloat16)
        return inputs


@pytest.mark.parametrize(
    "mode",
    ["train", "eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_RMBG(record_property, mode, op_by_op):
    if mode == "train":
        pytest.skip()
    model_name = "RMBG"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    tester = ThisTester(
        model_name, mode, compiler_config=cc, record_property_handle=record_property
    )

    with torch.no_grad():
        results = tester.test_model()
    if mode == "eval":
        predictions = results[-1].sigmoid()
        pred = predictions[0].squeeze()
        pred = pred.to(torch.float32)
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(tester.image.size)
        tester.image.putalpha(mask)
        tester.image.save("no_bg_image.png")

    tester.finalize()
