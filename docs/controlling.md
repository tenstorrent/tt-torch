## Controlling Compiler Behaviour

You can use the following environment variables to override default behaviour:

| Environment Variable | Behaviour | Default |
| -------------------- | --------- | --------
| TT_TORCH_COMPILE_DEPTH | Sets the maximum compile depth, see `tt_torch/tools/utils.py` for options. | `EXECUTE` |
| TT_TORCH_VERIFY_INTERMEDIATES | Sets whether to verify intermediate tensors against pytorch when running with compile depth `EXECUTE_OP_BY_OP`. | False |
| TT_TORCH_CONSTEVAL | Enables evaluation of constant expressions (consteval) in the Torch FX graph prior to compilation. | False |
| TT_TORCH_CONSTEVAL_PARAMETERS | Extends consteval to include parameters (e.g., model weights) as well as embedded constants. | False |
| TT_TORCH_EMBEDDEDD_CONSTANTS | Remove embedded constants from the Torch FX graph and convert them to constant inputs | False |
| TT_TORCH_IR_LOG_LEVEL | Enables printing MLIR from Torch to TTNN. It supports two modes; `INFO` and `DEBUG`. `INFO` prints MLIR for all conversions steps (Torch, StableHLO, TTIR and TTNN MLIR graphs). `DEBUG` prints intermediate MLIR for all passes (IR dump before and after each pass) additionally. Be warned, `DEBUG` IR printing forces single core compile, so it is much slower. | Disable |
