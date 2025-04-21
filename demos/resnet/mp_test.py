# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import backend
from PIL import Image
import torchvision.models as models
import tabulate
import requests
from tt_torch.tools.device_manager import DeviceManager
import time
import pprint
import threading
import torch.multiprocessing as mp
try:
   mp.set_start_method('fork', force=True)
   print("MP Start method set to fork")
except RuntimeError:
   pass

class CustomProcess(mp.Process):
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
    print("Downloading image from URL: ", url)
    img = Image.open(requests.get(url, stream=True).raw)
    print("Preprocessing image")
    img = preprocess(img)
    print("Unsqueezing image")
    img = torch.unsqueeze(img, 0)
    print("Converting image to bfloat16")
    img = img.to(torch.bfloat16)
    return img

def worker(idx, urls, topk=5):
    print("worker process ID: ", mp.current_process().pid)
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    parent = list(DeviceManager.get_parent_devices())[0]
    device = DeviceManager.get_sub_mesh_devices(parent)[idx]
    print("Using device: ", device)
    multi_options = {}
    multi_options["compiler_config"] = cc
    multi_options["device"] = device
    # Compile the model for each device
    tt_model = torch.compile(model, backend=backend, dynamic=False, options=multi_options)

    results = []
    headers = [f"Top {topk} Predictions"]
    for url in urls:
        print("Processing URL: ", url)
        img = download_image(url)
        print("Image downloaded")
        top5, top5_indices = torch.topk(tt_model(img).squeeze().softmax(-1), topk)
        print("Top 5 predictions obtained")
        tt_classes = []
        for class_likelihood, class_idx in zip(top5.tolist(), top5_indices.tolist()):
            print("Appending class likelihood and index")
            tt_classes.append(f"{classes[class_idx]}: {class_likelihood}")
        print("Class likelihoods and indices appended")
        rows = []
        url_string = f"Image URL: {url}\n"
        for i in range(topk):
            rows.append([tt_classes[i]])
        results.append(url_string + tabulate.tabulate(rows, headers=headers))
    return results


def multidevice(divided_urls):
    print("[multidevice fn] process ID: ", mp.current_process().pid)
    num_devices = len(divided_urls)
    processes = []
    for i in range(num_devices):
        p = CustomProcess(target=worker, args=(i, divided_urls[i]))
        processes.append(p)
        p.start()

    results = []
    for p in processes:
        result = p.join()
        results.append(result)
    return results
    # with mp.Pool(processes=num_devices) as pool:
    #     results = pool.starmap(worker, [(i, divided_urls[i]) for i in range(num_devices)])
    # return results

def main():

    NUM_ITERATIONS = 1  # Number of perf testing iterations to run

    # Prepare input urls
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
    num_devices = DeviceManager.get_num_available_devices()
    k, m = divmod(len(image_urls), num_devices)
    divided_urls = [
        image_urls[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(num_devices)
    ]
    
    # Compile model for each device
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    parent, _ = DeviceManager.acquire_available_devices()

    print("Executing dummy inference on multidevice to warm up the devices")
    # compile and execute this once to get the compilation overhead out of the way
    multidevice([[divided_urls[0][0]], [divided_urls[1][0]]])
    print("Dummy inference complete")

    print("Executing inference on multidevice")
    acc_duration = 0
    for _ in range(NUM_ITERATIONS):
        start_time = time.time()
        multi_results = multidevice(divided_urls)
        end_time = time.time()
        acc_duration += end_time - start_time
    avg_duration_multi = acc_duration / NUM_ITERATIONS

    print("Average inference time for multidevice: ", avg_duration_multi)
    print("Multi Device Results from latest iteration:")
    pprint.pprint(multi_results)

    DeviceManager.release_parent_device(parent, cleanup_sub_devices=True)

    
    

if __name__ == "__main__":
    print("Main process ID: ", mp.current_process().pid)
    print("MP start method:", mp.get_start_method())
    main()
