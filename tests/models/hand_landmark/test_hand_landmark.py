# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

import os
import subprocess
import sys
from pathlib import Path
import pytest
from tests.utils import ModelTester
from tt_torch.tools.utils import OpByOpBackend
from third_party.tt_forge_models.tools.utils import get_file

dependencies = ["mediapipe"]


class ThisTester(ModelTester):
    def _load_model(self):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_asset_path = Path(__file__).parent / "hand_landmarker.task"

        base_options = python.BaseOptions(model_asset_path=str(model_asset_path))
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        detector = vision.HandLandmarker.create_from_options(options)
        return detector.detect

    def _load_inputs(self):
        import mediapipe as mp

        image_file = get_file(
            "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg"
        )
        image = mp.Image.create_from_file(str(image_file))
        return image


@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.xfail(reason="Need to debud")
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_hand_landmark(record_property, mode, op_by_op):
    model_name = "Hand Landmark"

    # Download required files unless they already exist
    urls = [
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    ]
    for file_url in urls:
        os.system(f"wget -P {Path(__file__).parent} -nc {file_url}")

    """
    Workaround!
    -----------
    We decided to install the Python package below within the test, rather than
    using the typical approach with the `dependencies` variable mentioned above.
    The reason is that we want to overwrite the dependencies installed by the
    standard method and ensure we are using the exact package specified below.
    If we don't, we may unintentionally use a package that requires GPU support.

    Here's the background: this test uses the Hand Landmark model from the `mediapipe`
    package, which we need to install. However, installing this package also
    pulls in a dependent package, `opencv-python`, which unfortunately requires
    GPU support. Fortunately, we found a lightweight alternative,
    `opencv-python-headless`, that does not require GPU support. Since we can't
    prevent the installation of the undesired package, we install the preferred
    one afterward to ensure it is being used. This is the most efficient
    workaround I can think of.
    """
    # Uninstall the GPU version of opencv packages.
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "opencv-python",
            "opencv-contrib-python",
        ]
    )
    # Install the CPU version of opencv package.
    # Need `--force-reinstall` to handle the case that this CPU opencv has been
    # installed before this test, which leads to the ignorance of the following
    # installation if without `--force-reinstall`.
    # However, this package actually becomes broken and needs reinstallation
    # because some of the common dependencies between GPU and CPU opencv were
    # uninstalled by the above uninstallation of GPU opencv.
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "opencv-python-headless==4.8.0.74",
        ]
    )

    tester = ThisTester(model_name, mode, record_property_handle=record_property)
    tester.test_model()
    tester.finalize()
