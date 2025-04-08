# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import backend
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import tabulate
from threading import Thread
import requests
from tt_torch.tools.device_manager import DeviceManager
import time
import pprint

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


def get_predictions(tt_model, urls, topk=5):
    """
    Given the compiled tt_model and a list of URLs, this function calls the model
    for each URL and returns the top K predictions.
    """
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
    return results


def multidevice(divided_urls):
    """
    Running the ResNet model on all available devices in parallel.
    """
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    options = {}
    options["compiler_config"] = cc
    devices = DeviceManager.get_available_devices()  # Acquire all available devices
    num_devices = len(devices)
    tt_models = []
    for device in devices:
        options["device"] = device
        # Compile the model for each device
        tt_models.append(
            torch.compile(model, backend=backend, dynamic=False, options=options)
        )

    threads = []
    for i in range(num_devices):
        thread = CustomThread(
            target=get_predictions, args=(tt_models[i], divided_urls[i])
        )
        threads.append(thread)

    for thread in threads:
        thread.start()

    final_results = []
    for thread in threads:
        predictions = thread.join()
        final_results.append(predictions)
    DeviceManager.release_devices()  # Release all devices after use
    return final_results


def singledevice(image_urls):
    """
    Running the ResNet model on a single device.
    """
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    options = {}
    options["compiler_config"] = cc
    tt_model = torch.compile(model, backend=backend, dynamic=False, options=options)

    return get_predictions(tt_model, image_urls)


if __name__ == "__main__":
    NUM_ITERATIONS = 10  # Number of perf testing iterations to run
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

    # Prepare the input for the multidevice testing function.
    # This creates a list of lists of length num_devices, where the ith sublist
    # contains the image URLs that will be processed by the ith device.
    num_devices = DeviceManager.get_num_available_devices()
    k, m = divmod(len(image_urls), num_devices)
    divided_urls = [
        image_urls[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(num_devices)
    ]

    single_results = None
    multi_results = None
    acc_duration = 0
    for _ in range(NUM_ITERATIONS):
        start_time = time.time()
        single_results = singledevice(image_urls)
        end_time = time.time()
        acc_duration += end_time - start_time
    avg_duration_single = acc_duration / NUM_ITERATIONS
    acc_duration = 0
    for _ in range(NUM_ITERATIONS):
        start_time = time.time()
        multi_results = multidevice(divided_urls)
        end_time = time.time()
        acc_duration += end_time - start_time
    avg_duration_multi = acc_duration / NUM_ITERATIONS
    print("\n" * 5)
    print("Average Single device time (in s): ", avg_duration_single)
    print("Average Multi device time (in s): ", avg_duration_multi)
    print("\n" * 5)

    print("Single Device Results from latest iteration:")
    pprint.pprint(single_results)
    print("*" * 50)
    print("Multi Device Results from latest iteration:")
    pprint.pprint(multi_results)
