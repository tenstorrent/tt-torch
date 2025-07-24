# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import tt_mlir
import torch
import tt_torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import BackendOptions
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import tabulate
import requests
from tt_torch.tools.device_manager import DeviceManager


def main(num_loops):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights).to(torch.bfloat16).eval()
    classes = weights.meta["categories"]
    preprocess = weights.transforms()

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.enable_optimizer = True
    cc.enable_program_cache = True
    cc.enable_trace = True
    cc.trace_region_size = int(4 * 1e6)

    options = BackendOptions()
    options.compiler_config = cc
    tt_model = torch.compile(model, backend="tt", dynamic=False, options=options)

    headers = ["Top 5 Predictions"]
    topk = 5

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    def process_image():
        img = Image.open(requests.get(url, stream=True).raw)
        img = preprocess(img)
        img = torch.stack([img] * 8, dim=0)
        img = img.to(torch.bfloat16)

        outputs = tt_model(img)
        first_output = outputs[0]
        top5, top5_indices = torch.topk(first_output.squeeze().softmax(-1), topk)
        tt_classes = []
        for class_likelihood, class_idx in zip(top5.tolist(), top5_indices.tolist()):
            tt_classes.append(f"{classes[class_idx]}: {class_likelihood}")

        rows = []
        for i in range(topk):
            rows.append([tt_classes[i]])

        print(tabulate.tabulate(rows, headers=headers))
        print()

    print("Running with image: ", url)
    for i in range(num_loops):
        print("*" * 100)
        print(f"Running loop {i+1}/{num_loops}")
        process_image()


if __name__ == "__main__":
    tt_mlir.enable_persistent_kernel_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Number of times to run the demo.",
    )
    args = parser.parse_args()
    main(args.loops)
