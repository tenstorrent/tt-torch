# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import backend
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import tabulate
import requests
import time
import tt_mlir
from tt_torch.tools.device_manager import DeviceManager

weights = models.ResNet152_Weights.IMAGENET1K_V2
model = models.resnet152(weights=weights).to(torch.bfloat16).eval()
classes = weights.meta["categories"]
preprocess = weights.transforms()


def download_image(url):
    """
    Helper function to download an image from a URL and preprocess it.
    """
    img = Image.open(requests.get(url, stream=True).raw)
    img = preprocess(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(torch.bfloat16)
    return img


def main():
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    options = {}
    options["compiler_config"] = cc
    tt_model = torch.compile(model, backend=backend, dynamic=False, options=options)

    # List of image URLs to be processed
    image_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # Two cats
        "https://farm5.staticflickr.com/4106/4962771032_82d3b7ccea_z.jpg",  # Two zebras
        "https://farm5.staticflickr.com/4039/4184303499_115369327f_z.jpg",  # Pizza
        "https://farm5.staticflickr.com/4117/4902338213_9c6fb559b8_z.jpg",  # Park bench
        "https://farm4.staticflickr.com/3744/10085008474_8d72a9dc5e_z.jpg",  # Locomotive
        "https://farm4.staticflickr.com/3596/3687601495_73a46536b8_z.jpg",  # Baseball player
        "https://farm2.staticflickr.com/1375/5163062341_fbeb2e6678_z.jpg",  # Person in suit
        "https://farm2.staticflickr.com/1366/976992600_3927559756_z.jpg",  # Microwave
        "https://farm6.staticflickr.com/5056/5457805814_df70ed85c3_z.jpg",  # Labrador retriever
        "https://farm8.staticflickr.com/7325/9536735356_c1e2e5a0d5_z.jpg",  # Two elephants
    ]

    images = [download_image(url) for url in image_urls]

    print("Starting fake inference")
    tt_model(images[0])
    print("Done fake inference")

    results = []
    headers = ["Top 5 Predictions"]
    # start_time = time.time()
    for i in range(len(images)):
        img = images[i]
        url = image_urls[i]
        start = time.time()
        torch_tensor = tt_model(img)
        end = time.time()
        print(f"[SYNC DEBUG] time taken for torch tensors: {end - start:.5f} seconds")
        top5, top5_indices = torch.topk(torch_tensor.squeeze().softmax(-1), 5)
        tt_classes = []
        for class_likelihood, class_idx in zip(top5.tolist(), top5_indices.tolist()):
            tt_classes.append(f"{classes[class_idx]}: {class_likelihood}")
        rows = []
        url_string = f"Image URL: {url}\n"
        for i in range(5):
            rows.append([tt_classes[i]])
        results.append(url_string + tabulate.tabulate(rows, headers=headers))
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print()
    # print(f"Total time taken for inference: {elapsed_time:.2f} seconds")
    # print()
    for result in results:
        print("*" * 40)
        print(result)
        print()
        print("*" * 40)


if __name__ == "__main__":
    main()
