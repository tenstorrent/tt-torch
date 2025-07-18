# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# We must import tt_torch here as its import will register tenstorrents PJRT plugin with torch-xla.
import tt_torch

import torch
import argparse
import torchvision.models as models
import tabulate
import requests
from PIL import Image


def main(run_interactive):

    # Retrieve model
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights).to(torch.bfloat16).eval()
    classes = weights.meta["categories"]
    preprocess = weights.transforms()

    # Move model to TT device
    model = model.to("xla")

    headers = ["Top 5 Predictions"]
    topk = 5

    DEFAULT_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

    def process_image(img_path):
        if img_path == "":
            url = DEFAULT_URL
            img = Image.open(requests.get(url, stream=True).raw)
        elif img_path.startswith("http"):
            img = Image.open(requests.get(img_path, stream=True).raw)
        else:
            img = Image.open(img_path)
        img = preprocess(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(torch.bfloat16)

        # Move input activations to device
        img = img.to("xla")

        # Run model
        output = model(img).squeeze().softmax(-1)

        # torch-xla will compile all operations performed on a tensor until
        # it must move back to host. We do not currently support topk through
        # the compiler so we will move it back to host.
        output = output.to("cpu")

        top5, top5_indices = torch.topk(output, topk)
        tt_classes = []
        for class_likelihood, class_idx in zip(top5.tolist(), top5_indices.tolist()):
            tt_classes.append(f"{classes[class_idx]}: {class_likelihood}")

        rows = []
        for i in range(topk):
            rows.append([tt_classes[i]])

        print(tabulate.tabulate(rows, headers=headers))
        print()

    if not run_interactive:
        print("Running with default image image: ", DEFAULT_URL)
        process_image("")
    else:
        prompt = 'Enter the path of the image (type "stop" to exit or hit enter to use a default image): '
        img_path = input(prompt)
        while img_path != "stop":
            process_image(img_path)
            img_path = input(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_interactive",
        action="store_true",
        help="Run the demo interactively as opposed to once with the default image.",
    )
    args = parser.parse_args()
    main(args.run_interactive)
