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


def get_runtime_tensors(tt_model, imgs):
    runtime_tensors_list = []
    for img in imgs:
        runtime_tensors_list.append(tt_model(img))
    return runtime_tensors_list


def get_final_results(runtime_tensors_list):
    results = []
    headers = ["Top 5 Predictions"]
    for runtime_tensors in runtime_tensors_list:
        torch_tensor = tt_mlir.to_host(runtime_tensors)
        top5, top5_indices = torch.topk(torch_tensor.squeeze().softmax(-1), 5)
        tt_classes = []
        for class_likelihood, class_idx in zip(top5.tolist(), top5_indices.tolist()):
            tt_classes.append(f"{classes[class_idx]}: {class_likelihood}")
        rows = []
        url_string = f"Image URL: {url}\n"
        for i in range(5):
            rows.append([tt_classes[i]])
        results.append(url_string + tabulate.tabulate(rows, headers=headers))
    return results


def main():
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    num_devices = DeviceManager.get_num_available_devices()
    parent, devices = DeviceManager.acquire_available_devices(enable_async_ttnn=True)

    tt_models = []
    for device in devices:
        options = {}
        options["compiler_config"] = cc
        options["device"] = device
        options["async_mode"] = True
        tt_models.append(
            torch.compile(model, backend=backend, dynamic=False, options=options)
        )

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
    # Evenly distribute the image URLs across all devices.
    # This creates a list of lists of length num_devices, where the ith sublist
    # contains the image URLs that will be processed by the ith device.
    k, m = divmod(len(image_urls), num_devices)
    # divided_urls = [
    #     image_urls[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
    #     for i in range(num_devices)
    # ]
    divided_images = [
        images[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(num_devices)
    ]

    final_runtime_tensors_list = []

    threads = []
    for i in range(num_devices):
        thread = CustomThread(
            target=get_runtime_tensors, args=(tt_models[i], divided_images[i])
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        final_runtime_tensors_list.append(thread.join())

    start_time = time.time()
    for i in range(num_devices):
        tt_model = tt_models[i]
        imgs = divided_images[i]
        for img in imgs:
            # img = download_image(url)
            runtime_tensors_list.append(tt_model(img))

    results = []
    headers = ["Top 5 Predictions"]
    for i in range(len(runtime_tensors_list)):
        runtime_tensors = runtime_tensors_list[i]
        url = image_urls[i]
        torch_tensor = tt_mlir.to_host(runtime_tensors)
        top5, top5_indices = torch.topk(torch_tensor.squeeze().softmax(-1), 5)
        tt_classes = []
        for class_likelihood, class_idx in zip(top5.tolist(), top5_indices.tolist()):
            tt_classes.append(f"{classes[class_idx]}: {class_likelihood}")
        rows = []
        url_string = f"Image URL: {url}\n"
        for i in range(5):
            rows.append([tt_classes[i]])
        results.append(url_string + tabulate.tabulate(rows, headers=headers))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print()
    print(f"Total time taken for inference: {elapsed_time:.2f} seconds")
    print()
    for result in results:
        print("*" * 40)
        print(result)
        print()
        print("*" * 40)

    DeviceManager.release_parent_device(parent, cleanup_sub_devices=True)


if __name__ == "__main__":
    main()
