# How to use
Build/install and activate the environment as you would for development on tt-torch: [Build Instructions](https://docs.tenstorrent.com/tt-torch/build.html)

From the project root, run
```
python demos/resnet/resnet50_demo.py
```
for the interactive demo that uses a single-device to classify an image using ResNet.

Or run,
```
python demos/resnet/resnet50_data_parallel_demo.py [--use_simplified_manager]
```
for the demo on how to leverage the `DeviceManager` module to run ResNet image classification across multiple chips in parallel.

# Results
## resnet50_demo.py
When running this on the default image (http://images.cocodataset.org/val2017/000000039769.jpg):

![Image of two cats](http://images.cocodataset.org/val2017/000000039769.jpg)

The output is ResNet's top 5 predictions:
```
Top 5 Predictions
----------------------------------
tabby: 0.56640625
tiger cat: 0.236328125
Egyptian cat: 0.02734375
pillow: 0.005218505859375
remote control: 0.0037689208984375
```
Notice that in the code there was no explicit device management:
```Python
options = {}
options["compiler_config"] = cc
# We didn't provide any explicit device while
# compiling the model.
tt_model = torch.compile(model, backend=backend, dynamic=False, options=options)
```
This causes the model to be compiled onto the default device present in the board. The device acquisition and release get handled automatically.

## resnet50_data_parallel_demo.py
This file shows how to split multiple model requests across all available devices on the board using the `DeviceManager` module. The relevant device management code is:

```Python
from tt_torch.tools.device_manager import DeviceManager
...
def main():
    ...
    options = {}
    options["compiler_config"] = cc
    # Acquires all available devices and returns them in a list
    parent, devices = DeviceManager.acquire_available_devices()
    tt_models = []
    for device in devices:
        options["device"] = device # Explicitly compile the model for a specific device
        tt_models.append(
            torch.compile(model, backend=backend, dynamic=False, options=options)
        )
    ...
    # Release all acquired devices after use
    DeviceManager.release_parent_device(parent, cleanup_sub_devices=True)
```
Note that in this case, the user is responsible for releasing the acquired devices.

The file defines 10 Image URLs, which get split into N groups of 10/N URLs (where N is the number of available devices on the board).

Each device will then call the relevant compiled model to independently process the assigned grouping in parallel.

When ran on an N300 board with 2 available devices the results are:

### First device:
```
****************************************
Results from Device: <tt_mlir.Device object at 0x7fc4e283d1b0>
Image URL: http://images.cocodataset.org/val2017/000000039769.jpg
Top 5 Predictions
----------------------------------
tabby: 0.56640625
tiger cat: 0.236328125
Egyptian cat: 0.02734375
pillow: 0.005218505859375
remote control: 0.0037689208984375

Image URL: https://farm5.staticflickr.com/4106/4962771032_82d3b7ccea_z.jpg
Top 5 Predictions
----------------------------------
zebra: 0.85546875
ostrich: 0.00115203857421875
hartebeest: 0.00067138671875
Arabian camel: 0.00046539306640625
cheetah: 0.000400543212890625
.
.
.
{3 other URL results}
****************************************
```
### Second device:
```
****************************************
Results from Device: <tt_mlir.Device object at 0x7fc400e79f70>
Image URL: https://farm4.staticflickr.com/3596/3687601495_73a46536b8_z.jpg
Top 5 Predictions
------------------------------
ballplayer: 0.77734375
baseball: 0.06787109375
racket: 0.0028533935546875
scoreboard: 0.0011138916015625
collie: 0.000713348388671875

Image URL: https://farm2.staticflickr.com/1375/5163062341_fbeb2e6678_z.jpg
Top 5 Predictions
----------------------------
suit: 0.703125
spotlight: 0.07861328125
Windsor tie: 0.0155029296875
groom: 0.005889892578125
notebook: 0.005523681640625
.
.
.
{3 other URL results}
****************************************
```
