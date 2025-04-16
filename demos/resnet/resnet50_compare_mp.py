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
from threading import Thread
import requests
from tt_torch.tools.device_manager import DeviceManager
import time
import pprint
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

global tt_models
tt_models = [1]


# A custom thread class to simplify returning values from threads
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
    img = Image.open(requests.get(url, stream=True).raw)
    img = preprocess(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(torch.bfloat16)
    return img


def get_predictions(model_index, urls, topk=5):
    """
    Given the compiled tt_model and a list of URLs, this function calls the model
    for each URL and returns the top K predictions.
    """
    global tt_models
    results = []
    print("MODELS: ", tt_models)
    print("MODEL INDEX: ", model_index)
    tt_model = tt_models[model_index]
    print("GET PREDICTIONS CALLED")
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

def setup(rank, world_size):
    # os.environ["MASTER_ADDR"] = "127.0.0.1"
    # os.environ["MASTER_PORT"] = "43009"
    print("SETUP CALLED")
    dist.init_process_group("gloo", init_method="file:///d:/tmp/store_file", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    print("CLEANUP CALLED")

def worker(rank, world_size, tt_models, divided_urls):
    setup(rank, world_size)
    tt_models[rank] = DDP(tt_models[rank], device_ids=None, output_device=None).module
    results = get_predictions(tt_models[rank], divided_urls[rank])
    cleanup()
    return results

def multidevice(divided_urls):
    """
    Running the ResNet model on all available devices in parallel.
    """
    global tt_models
    
    # processes = []
    # for i in range(num_devices):
    #     tt_models[i] = DDP(tt_models[i], device_ids=None, output_device=None).module
    # mp.spawn(
    #     worker,
    #     args=(num_devices, tt_models, divided_urls),
    #     nprocs=num_devices,
    # )
    # for i in range(num_devices):
    #     setup(i, num_devices)
    #     print("FINISHED SETUP")
    #     tt_models[i] = DDP(tt_models[i], device_ids=None, output_device=None).module
    
    # with mp.Manager() as manager:
    #     tt_models = manager.list(tt_models)
    #     num_devices = len(tt_models)
    #     print("MULTIDEVICE NUM DEVICES: ", num_devices)
    #     with mp.Pool(processes=num_devices) as pool:
    #         final_results = pool.starmap(get_predictions, zip(range(num_devices), divided_urls))
    #     return final_results

    with mp.Pool(processes=num_devices) as pool:
        final_results = pool.starmap(get_predictions, zip(range(num_devices), divided_urls))
    return final_results

    # for i in range(num_devices):
    #     setup(i, num_devices)
    #     tt_model = DDP(tt_models[i], device_ids=None, output_device=None)
    #     p = CustomProcess(target=get_predictions, args=(tt_model.module, divided_urls[i]))
    #     processes.append(p)
    #     p.start()
    
    # final_results = []
    # for p in processes:
    #     results = p.join()
    #     final_results.append(results)
    # return final_results
    
    



def singledevice(tt_model, image_urls):
    """
    Running the ResNet model on a single device.
    """
    return get_predictions(tt_model, image_urls)


if __name__ == "__main__":
    NUM_ITERATIONS = 1  # Number of perf testing iterations to run
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

    # Compile for single and multi-device
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    single_options = {}
    single_options["compiler_config"] = cc
    single_model = torch.compile(model, backend=backend, dynamic=False, options=single_options)
    
    # print("Executing dummy inference on single device to warm up the devices")
    # # compile and execute this once to get the compilation overhead out of the way
    # singledevice(single_model, image_urls[:2])
    # print("Compiled single device model")
    # single_results = None
    # acc_duration = 0
    # print("Testing single-device performance")
    # for _ in range(NUM_ITERATIONS):
    #     start_time = time.time()
    #     single_results = singledevice(single_model, image_urls)
    #     end_time = time.time()
    #     acc_duration += end_time - start_time
    # avg_duration_single = acc_duration / NUM_ITERATIONS

    parent, devices = DeviceManager.acquire_available_devices()
    num_devices = len(devices)

    for device in devices:
        multi_options = {}
        multi_options["compiler_config"] = cc
        multi_options["device"] = device
        # Compile the model for each device
        multi_model = torch.compile(model, backend=backend, dynamic=False, options=multi_options)
        tt_models.append(multi_model)

    print("Compiled multi device models")
    print("Executing dummy inference on multidevice to warm up the devices")
    # compile and execute this once to get the compilation overhead out of the way
    multidevice([[divided_urls[0][0]], [divided_urls[1][0]]])
    print("Dummy inference complete")

    # multi_results = None
    # acc_duration = 0
    # print("Testing multi-device performance")
    # for _ in range(NUM_ITERATIONS):
    #     start_time = time.time()
    #     multi_results = multidevice(tt_models, divided_urls)
    #     end_time = time.time()
    #     acc_duration += end_time - start_time
    # avg_duration_multi = acc_duration / NUM_ITERATIONS
    
    # DeviceManager.release_parent_device(parent, cleanup_sub_devices=True)
    
    # print("\n" * 5)
    # print("Average Multi device time (in s): ", avg_duration_multi)
    # # print("Average Single device time (in s): ", avg_duration_single)
    # print("\n" * 5)

    # print("Multi Device Results from latest iteration:")
    # pprint.pprint(multi_results)
    # print("*" * 50)
    # print("Single Device Results from latest iteration:")
    # # pprint.pprint(single_results)


# Look at how backend.py handles binary serialization.
# bindings.cpp has an example of how to convert binary to json and vice versa.