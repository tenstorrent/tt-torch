## Controlling Compiler Behaviour

You can use the following environment variables to override default behaviour:

| Environment Variable | Behaviour | Default |
| -------------------- | --------- | --------
| TT_TORCH_COMPILE_DEPTH | Sets the maximum compile depth, see `tt_torch/tools/utils.py` for options. | `EXECUTE` |
| TT_TORCH_VERIFY_OP_BY_OP | Sets whether to verify the output of each compiled op against pytorch when running with compile depth `EXECUTE_OP_BY_OP`. | False |
| TT_TORCH_VERIFY_INTERMEDIATES | Sets whether to verify runtime intermediates during execution. | False |
| TT_TORCH_CONSTEVAL | Enables evaluation of constant expressions (consteval) in the Torch FX graph prior to compilation. | False |
| TT_TORCH_CONSTEVAL_PARAMETERS | Extends consteval to include parameters (e.g., model weights) as well as embedded constants. | False |
| TT_TORCH_INLINE_PARAMETERS | Inlines parameters in the MLIR module (and thus flatbuffer executable) rather than requiring them as inputs. NOTE: The maximum size of a flatbuffer is 2GB so this will cause compilation to fail for sufficiently large models | False |
| TT_TORCH_IR_LOG_LEVEL | Enables printing MLIR from Torch to TTNN. It supports two modes; `INFO` and `DEBUG`. `INFO` prints MLIR for all conversions steps (Torch, StableHLO, TTIR and TTNN MLIR graphs). `DEBUG` prints intermediate MLIR for all passes (IR dump before and after each pass) additionally. Be warned, `DEBUG` IR printing forces single core compile, so it is much slower. | Disable |

### Controlling Compiler Behaviour Programatically

Instead of using the above environment variables, compiler behaviour can be configured programatically as well.

Here is an example of enabling consteval:
```py
from tt_torch.dynamo.backend import backend
from tt_torch.tools.utils import CompilerConfig
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        ...

    def foward(self, ...):
        ...

model = MyModel()

cc = CompilerConfig()
cc.enable_consteval = True
cc.consteval_parameters = True # This will enable constant folding on the parameters in addition to any constants

model = torch.compile(model, backend=backend, options=cc)

inputs = ...

outputs = model(inputs)
```
