# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from PIL import Image
from torchvision import transforms
import subprocess
import sys
import os
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file

dependencies = ["ultralytics==8.2.92", "ultralytics-thop==2.0.6"]


class ThisTester(ModelTester):
    def _load_model(self):
        """
        The model is from https://pytorch.org/hub/ultralytics_yolov5/
        """

        """
        Workaround!
        -----------
        We decided to install the Python package below within the test, rather than
        using the typical approach with the `dependencies` variable mentioned above.
        The reason is that we want to overwrite the dependencies installed by the
        standard method and ensure we are using the exact package specified below.
        If we don't, we may unintentionally use a package that requires GPU support.

        Here's the background: this test uses the YOLOv5 model from the `ultralytics`
        package, which we need to install. However, installing this package also
        pulls in a dependent package, `opencv-python`, which unfortunately requires
        GPU support. Fortunately, we found a lightweight alternative,
        `opencv-python-headless`, that does not require GPU support. Since we can't
        prevent the installation of the undesired package, we install the preferred
        one afterward to ensure it is being used. This is the most efficient
        workaround I can think of.
        """
        # # Uninstall the GPU version of opencv packages.
        # subprocess.check_call(
        #     [sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python"]
        # )
        # # Install the CPU version of opencv package.
        # # Need `--force-reinstall` to handle the case that this CPU opencv has been
        # # installed before this test, which leads to the ignorance of the following
        # # installation if without `--force-reinstall`.
        # # However, this package actually becomes broken and needs reinstallation
        # # because some of the common dependencies between GPU and CPU opencv were
        # # uninstalled by the above uninstallation of GPU opencv.
        # subprocess.check_call(
        #     [sys.executable, "-m", "pip", "install", "--force-reinstall", "opencv-python-headless==4.8.0.74"]
        # )

        # Model
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load(
            "ultralytics/yolov5",
            "yolov5s",
            pretrained=True,
            autoshape=False,
            device="cpu",
        )

        # Remove the downloaded pretrained weight file.
        """
        * Safe to remove: The weight file is an intermediate step of loading
            weights. Finally it has a copy in memory, and the file is no longer
            necessary.
        * Not in `.cache/`: Downloaded files like this one are supposed to store
            at directory `~/.cache/`, where temporary and downloaded files are
            collected together during testing, and we can easily do the cleaning
            after testing. Actually, function `torch.hub.load` you can see above
            is supposed to follow this convention. However, this test is an
            exception. The model YOLOv5 does not come directly from Torch Hub.
            In fact, it comes from another package `ultralytics`. Unfortunately,
            `ultralytics` does not follow the cache convention. It saves
            downloaded files at where the test is run, which is not good and may
            leave garbage files everywhere. Therefore, we decided to clean it by
            the test itself.
        """
        downloaded_file = "yolov5s.pt"
        if os.path.exists(downloaded_file):
            os.remove(downloaded_file)

        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python-headless"]
        )
        return model.to(torch.bfloat16)

    def _load_inputs(self):
        # Image preprocessing
        # Local cache of https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
        image_file = get_file("test_images/dog_1546x1213.jpg")
        image = Image.open(str(image_file))
        transform = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor()]
        )
        img_tensor = [transform(image).unsqueeze(0)]
        batch_tensor = torch.cat(img_tensor, dim=0)
        return batch_tensor.to(torch.bfloat16)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_yolov5(record_property, mode, op_by_op):
    model_name = "YOLOv5"

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
        required_atol=12,
        record_property_handle=record_property,
        # TODO Enable checking - https://github.com/tenstorrent/tt-torch/issues/490
        assert_pcc=False,
        assert_atol=False,
    )
    tester.test_model()
    tester.finalize()
