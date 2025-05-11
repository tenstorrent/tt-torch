# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import numpy as np
from PIL import Image
from tests.utils import ModelTester
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend

from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.table_rec import TableRecPredictor


class ThisTester(ModelTester):
    def __init__(self, *args, model_type="recognition", **kwargs):
        # Set model type for different surya models (recognition, detection, layout, table_recognition)
        self.model_type = model_type

        # Call parent constructor
        super(ThisTester, self).__init__(*args, **kwargs)

    def _load_model(self):
        # Create appropriate model based on model_type
        if self.model_type == "recognition":

            print("Loading recognition model...")
            import os
            os.environ["COMPILE_RECOGNITION"] = "true"  # Enable surya's own compilation

            from surya.settings import settings
            print(f"KCM RECOGNITION_STATIC_CACHE: {settings.RECOGNITION_STATIC_CACHE}")

            model = RecognitionPredictor()
            # Set to eval mode explicitly
            model.model.eval()

            return model.model
        elif self.model_type == "detection":        # This works!
            model = DetectionPredictor()
            model.model.eval()
            return model.model
        elif self.model_type == "layout":
            model = LayoutPredictor()
            model.model.eval()
            return model.model
        elif self.model_type == "table_recognition":
            model = TableRecPredictor()
            model.model.eval()
            return model.model
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _load_inputs(self):
        # Generate dummy input image data appropriate for each model type
        if self.model_type == "recognition":
            # For recognition model - expects a cropped text image
            # The model expects specific dimensions for attention to work correctly
            # Using 224x224 which is a common input size for vision transformer models
            batch_size = 1
            channels = 3

            # height = 32  # Common height for text line recognition
            # width = 256  # Variable width, but we'll use a reasonable size

            height = 224  # Standard height for many vision models including those with attention
            width = 224   # Standard width for vision transformer inputs
            
            # height = 64   # Many OCR models use shorter heights, but wider widths
            # width = 512   # Wider to accommodate text lines

            # Create tensor with specific size that's compatible with the attention mechanism
            # return torch.rand(batch_size, channels, height, width)

            # Creating a real PIL image with white background for text line recognition
            from PIL import Image, ImageDraw
            import numpy as np
            import os
            
            # # Create a blank white image with suitable dimensions for text lines
            # width, height = 320, 64
            # img = Image.new('RGB', (width, height), color=(255, 255, 255))
            
            # # Add a simple black rectangle (simulating text without needing fonts)
            # draw = ImageDraw.Draw(img)
            # draw.rectangle([(50, 20), (270, 40)], fill=(0, 0, 0))
            
            # # Save the test image to disk for inspection
            # test_img_dir = os.path.dirname(os.path.abspath(__file__))
            # test_img_path = os.path.join(test_img_dir, "test_recognition_image.png")
            # img.save(test_img_path)
            # print(f"Saved test recognition image to: {test_img_path}")
            
            # # Convert PIL Image to tensor the way the model would expect
            # img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

            # # Get the tensor and print some stats about it
            # print(f"Input shape: {img_tensor.shape}")
            # print(f"Input dtype: {img_tensor.dtype}")
            # print(f"Input device: {img_tensor.device}")
            
            # # Add batch dimension
            # return img_tensor.unsqueeze(0)  # Shape: [1, 3, 64, 320]


            # NEW

            # We need to create an image for the recognition model
            # Create a white image with some black text-like elements
            # The recognition model expects [batch_size, channels, height, width]
            
            # Create a PIL image with some text-like elements
            import random

            height = 384  # Increase from 224 to 384
            width = 384  # Increase from 224 to 384 (square aspect ratio)
            
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw some black rectangles to simulate text
            for i in range(5):
                x1 = random.randint(10, width - 100)
                y1 = random.randint(10 + i * (height // 6), 10 + (i + 1) * (height // 6) - 20)
                x2 = x1 + random.randint(50, 150)
                y2 = y1 + random.randint(5, 15)
                draw.rectangle([x1, y1, x2, y2], fill='black')
            
            # Save the image for debugging
            os.makedirs("debug", exist_ok=True)
            img.save("debug/sample_input.png")
            
            # Convert PIL Image to tensor the way the model would expect
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

            # Get the tensor and print some stats about it
            print(f"Input shape: {img_tensor.shape}")
            print(f"Input dtype: {img_tensor.dtype}")
            print(f"Input device: {img_tensor.device}")
            
            # Add batch dimension
            return img_tensor.unsqueeze(0)  # Shape: [1, 3, 384, 384]




        elif self.model_type == "detection":
            # For detection model - expects full document image
            # Model takes (batch_size, channels, height, width)
            batch_size = 1
            channels = 3
            height = 512
            width = 512

            return torch.rand(batch_size, channels, height, width)

        elif self.model_type == "layout":
            # For layout model - expects full document image
            # Similar to detection
            batch_size = 1
            channels = 3
            height = 512
            width = 512

            return torch.rand(batch_size, channels, height, width)

        elif self.model_type == "table_recognition":
            # For table recognition model - expects cropped table image
            batch_size = 1
            channels = 3
            height = 384
            width = 384

            return torch.rand(batch_size, channels, height, width)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")



@pytest.mark.parametrize(
    "model_type",
    ["recognition", "detection", "layout", "table_recognition"],
)
@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
def test_surya(record_property, model_type, mode, op_by_op):
    model_name = f"surya-{model_type}"

    cc = CompilerConfig()
    cc.enable_consteval = False
    cc.consteval_parameters = False

    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO

    # Higher tolerance for surya models
    required_atol = 0.05

    tester = ThisTester(
        model_name,
        mode,
        model_type=model_type,
        required_atol=required_atol,
        compiler_config=cc,
        record_property_handle=record_property,
        model_group="red",
    )

    results = tester.test_model()

    # Print output information
    print(
        f"""
    Model: {model_name}
    Input shape: {tester.inputs.shape}
    Output shape: {results.shape if hasattr(results, 'shape') else 'N/A'}
    """
    )

    tester.finalize()
