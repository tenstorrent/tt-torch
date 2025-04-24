# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import backend
from PIL import Image
import torchvision.models as models
import tabulate
from threading import Thread
import requests
from tt_torch.tools.device_manager import DeviceManager

# import torch.multiprocessing as mp

# mp.set_start_method("spawn", force=True)

weights = models.ResNet152_Weights.IMAGENET1K_V2
model = models.resnet152(weights=weights).to(torch.bfloat16).eval()
classes = weights.meta["categories"]
preprocess = weights.transforms()

# A custom thread class to simplify returning values from threads
class CustomThread(Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs={}, verbose=None
    ):
        super().__init__(group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return


def download_image(url):
    """
    Helper function to download an image from a URL and preprocess it.
    """
    img = Image.open(requests.get(url, stream=True).raw)
    img = preprocess(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(torch.bfloat16)
    return img


def worker(device, urls, topk=5):
    """
    Given a device and a list of URLs, this function compiles the model and calls it
    for each URL, returning the top K predictions.
    """
    # print(f"Worker ID: {mp.current_process().pid}" f" on Device: {device}")
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    options = {}
    options["compiler_config"] = cc
    options["device"] = device
    tt_model = torch.compile(model, backend=backend, dynamic=False, options=options)

    results = []
    headers = [f"Top {topk} Predictions"]
    for url in urls:
        img = download_image(url)
        top5, top5_indices = torch.topk(tt_model(img).squeeze().softmax(-1), topk)
        tt_classes = []
        for class_likelihood, class_idx in zip(top5.tolist(), top5_indices.tolist()):
            tt_classes.append(f"{classes[class_idx]}: {class_likelihood}")
        rows = []
        url_string = f"Image URL: {url}\n"
        for i in range(topk):
            rows.append([tt_classes[i]])
        results.append(url_string + tabulate.tabulate(rows, headers=headers))
    return device, results


def main(use_simplified_manager, enable_async):
    print("Using async ttnn: ", enable_async)
    num_devices = DeviceManager.get_num_available_devices()
    if use_simplified_manager:
        print("Using simplified device manager")
        parent, devices = DeviceManager.acquire_available_devices(
            enable_async_ttnn=enable_async
        )
    else:
        print("Using full device manager")
        parent = DeviceManager.create_parent_mesh_device(
            mesh_shape=[1, num_devices], enable_async_ttnn=enable_async
        )
        for i in range(num_devices):
            DeviceManager.create_sub_mesh_device(
                parent, mesh_offset=(0, i), mesh_shape=[1, 1]
            )
        devices = list(DeviceManager.get_sub_mesh_devices(parent))
    print("Devices acquired: ", devices)

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
    # Evenly distribute the image URLs across all devices.
    # This creates a list of lists of length num_devices, where the ith sublist
    # contains the image URLs that will be processed by the ith device.
    k, m = divmod(len(image_urls), num_devices)
    divided_urls = [
        image_urls[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(num_devices)
    ]

    final_results = []
    if not enable_async:
        for i in range(num_devices):
            results = worker(devices[i], divided_urls[i])
            final_results.append(results)
    else:
        # Multi-threaded
        threads = []
        for i in range(num_devices):
            thread = CustomThread(target=worker, args=(devices[i], divided_urls[i]))
            threads.append(thread)
            thread.start()

        for thread in threads:
            result = thread.join()
            final_results.append(result)

    # Print the results
    for device_used, prediction_results in final_results:
        print("*" * 40)
        print(f"Results from Device: {device_used}")
        for result in prediction_results:
            print(result)
            print()
        print("*" * 40)

    if use_simplified_manager:
        # The cleanup_sub_devices flag will call the release_sub_mesh_device() method
        # for each sub mesh under the specified parent device automatically.
        DeviceManager.release_parent_device(parent, cleanup_sub_devices=True)
    else:
        # The option to manually release the sub mesh devices is also available.
        for device in devices:
            DeviceManager.release_sub_mesh_device(device)
        DeviceManager.release_parent_device(parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_simplified_manager",
        action="store_true",
        help="Use simplified device manager to run file",
    )
    parser.add_argument(
        "--enable_async",
        action="store_true",
        help="Enable async ttnn execution",
    )
    args = parser.parse_args()
    main(args.use_simplified_manager, args.enable_async)
