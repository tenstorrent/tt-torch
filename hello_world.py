from tt_torch.dynamo.backend import backend, BackendOptions
from tt_torch.tools.utils import CompilerConfig, CompileDepth
from tt_torch.tools.device_manager import DeviceManager

import torch

class AddTensors(torch.nn.Module):
  def forward(self, x, y):
    return x + y

model = AddTensors()
cc = CompilerConfig()
cc.compile_depth = CompileDepth.EXECUTE_CPP
cc.dump_info = True

num_devices = DeviceManager.get_num_available_devices()
print("Num devices available:", num_devices)
_, devices = DeviceManager.acquire_available_devices()
assert len(devices) == 1, "Only one device is supported for this example"

device = devices[0]
options = BackendOptions()
options.compiler_config = cc
options.async_mode = True
options.devices = [device]
tt_model = torch.compile(model, backend=backend, dynamic=False, options=options)

x = torch.ones(32, 32)
y = torch.ones(32, 32)
print(tt_model(x, y))
