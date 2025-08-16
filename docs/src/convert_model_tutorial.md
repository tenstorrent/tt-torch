# Convert a Model from Hugging Face for Use with TT-Torch

This walkthrough shows you how to take a model from HuggingFace and adapt it for use with TT-Torch and a Tenstorrent chip. The walkthrough uses the ConvNeXT Tiny model that can be found here: [convnext-tiny-224](https://huggingface.co/facebook/convnext-tiny-224).

ConvNeXT is a family of deep learning models designed to help with computer vision tasks such as image classification, object detection, medical imaging analysis, and autonomous driving systems. It is built on the ResNet architecture and incorporates design elements from Vision Transformers that allow for high accuracy, efficiency, and scalability. ConvNeXt-Tiny is a smaller, more lightweight version of ConvNeXt, making it easier to work with for a walkthrough. 

## Prerequisites

For this walkthrough you need: 

* [Getting Started Instructions](getting_started.md) - This shows you how to configure your hardware if you have not done so, and how to download the TT-Torch wheel. You do NOT need to install all the listed Python packages in the Getting Started instructions, only **pillow**. At the end of the installation you should have: 
    * Configured hardware
    * TT-Torch wheel 
    * TT-Forge repo 
* The Python package **pillow**

## Convert HuggingFace Code Example to Work with Tenstorrent

This section provides the original code from [HuggingFace for **convnext-tiny-224**](https://huggingface.co/facebook/convnext-tiny-224) and shows you what is required to convert it to work with TT-Torch. 

### ConvNeXT Tiny-Sized Model (convnext-tiny-224) From HuggingFace

Here is the code that needs to be converted to work with TT-Torch: 

```python
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224")
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label]),
```

Because most of this code is HuggingFace specific, converting it for use with TT-Torch involves replacing the use of: 
* ConvNextImageProcessor 
* ConvNextForImageClassification
* Datasets 

### Changing the HuggingFace Code to Work with Tenstorrent
To convert the HuggingFace example to work with Tenstorrent, the following changes need to happen in the code: 

| Hugging Face Component | Tenstorrent Component |
|------------------------|-----------------------|
| `ConvNextImageProcessor.from_pretrained(...)` | `weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 + weights.transforms()` |
| `ConvNextForImageClassification.from_pretrained(...)` | `model = convnext_tiny(weights=weights).eval()` |
| `load_dataset(...) and image = dataset["test"]["image"][0]` | `image = Image.open("000000039769.jpg")` | 
| `model(**inputs)` | `tt_model(input_tensor)` |
| `model.config.id2label[...]` | `weights.meta["categories"][...]` |

You also need `tt_model = torch.compile(model, backend="tt", dynamic=False, options=options)` to compile everything for use with a Tenstorrent chip.

To run this code for the walkthrough, an existing image is used. If you want to use your own image, you need to modify the code to prompt for a URL and open it, and you need to add **requests** to the list of imports. Otherwise, do the following: 

1. Navigate to **tt-forge/demos/tt-torch**. 

2. Create a python file and paste the code sample listed below into it:

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import BackendOptions
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision import transforms
from PIL import Image
import os

def main():
    # Path to the COCO image — ensure this file exists in your working dir
    image_path = "000000039769.jpg"
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    # Step 1: Open the image
    image = Image.open(image_path).convert("RGB")

    # Step 2: Load model weights and preprocessing pipeline
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    classes = weights.meta["categories"]

    # Step 3: Preprocess image
    input_tensor = preprocess(image).unsqueeze(0).to(torch.bfloat16)

    # Step 4: Load model
    model = convnext_tiny(weights=weights).eval().to(torch.bfloat16)

    # Step 5: Set up TT compile options
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    options = BackendOptions()
    options.compiler_config = cc

    # Step 6: Compile for Tenstorrent
    tt_model = torch.compile(model, backend="tt", dynamic=False, options=options)

    # Step 7: Run inference
    with torch.no_grad():
        logits = tt_model(input_tensor).squeeze()

    # Step 8: Print top-5 predictions
    probs = logits.softmax(dim=-1)
    top5_probs, top5_indices = torch.topk(probs, 5)

    print("\nTop 5 predictions:")
    for i in range(5):
        label = classes[top5_indices[i].item()]
        confidence = top5_probs[i].item() * 100
        print(f"{label:>25}: {confidence:.2f}%")


if __name__ == "__main__":
    main()
```

> **NOTE:** You place the file in **tt-forge/demos/tt-torch** so you can use the image **000000039769.jpg** that is provided in the folder. 

3. Save the file. 

4. Run the file. (python *FILE_NAME*)