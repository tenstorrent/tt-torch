tt-torch uses pytest for all unit and model tests.

Tests are organized into unit tests for pytorch (tests/torch), unit tests for onnx (test/onnx) and models (tests/models).
They can be run locally by running:

```bash
source env/activate
pytest -svv tests/torch
```


Model tests (tests/models) have the option to run op-by-op, see [overview](https://docs.tenstorrent.com/tt-torch/controlling.html). This allows for faster model bring-up as it allows users to find any potential issues in parallel. This is controlled by the `--op_by_op_torch` or `--op_by_op_stablehlo` flags. Example:

```bash
pytest -svv tests/models/albert --op_by_op_torch
```
