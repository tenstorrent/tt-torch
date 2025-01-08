# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import onnx
from onnx.tools import update_model_dims
import onnxruntime
import numpy as np
import collections
import re
from typing import List, Dict, Tuple
from tt_torch.dynamo.backend import backend
from tt_torch.onnx_compile import compile_onnx
from tt_torch.tools.utils import CompilerConfig, CompileDepth
import json
from onnx import version_converter
from pathlib import Path
from tt_torch.tools.verify import verify_against_golden


class ModelTester:
    def __init__(
        self,
        model_name,
        mode,
        pcc=0.99,
        required_atol=None,
        relative_atol=None,
        compiler_config=None,
    ):
        if mode not in ["train", "eval"]:
            raise ValueError(f"Current mode is not supported: {mode}")
        self.model_name = model_name
        self.mode = mode
        self.framework_model = self._load_model()
        self.compiled_model = None
        self.inputs = self._load_inputs()
        self.pcc = pcc

        if (required_atol is None) and (relative_atol is None):
            print(
                "Neither required_atol, or relative_atol is provided. Setting required_atol=0.01."
            )
            required_atol = 0.01

        assert (required_atol is not None) != (
            relative_atol is not None
        ), "Exactly one of required_atol or relative_atol should be provided."

        self.required_atol = required_atol
        self.relative_atol = relative_atol
        self.golden_outputs = None
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self.compiler_config = compiler_config
        self.compiler_config.model_name = model_name

    def _load_model(self):
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def _load_inputs(self):
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def _extract_outputs(self, output_object):
        if isinstance(output_object, torch.Tensor):
            return (output_object,)
        elif isinstance(output_object, (tuple, list)):
            is_only_tensors = True
            for item in output_object:
                if not isinstance(item, torch.Tensor):
                    is_only_tensors = False
                    break
            if is_only_tensors:
                return tuple(output_object)

        raise NotImplementedError(
            f"Output object type: ({type(output_object)}) is not a torch.Tensor, tuple[torch.Tensor], or list[torch.Tensor]. Please implement _extract_outputs in the derived class."
        )

    def get_golden_outputs(self, model, inputs):
        if self.golden_outputs is not None:
            return self.golden_outputs

        self.golden_outputs = self.run_model(model, inputs)
        return self.golden_outputs

    def compile_model(self, model, compiler_config):
        # Compile model
        model = torch.compile(
            model, backend=backend, dynamic=False, options=compiler_config
        )
        self.compiled_model = model
        return self.compiled_model

    def run_model(self, model, inputs):
        if isinstance(inputs, collections.abc.Mapping):
            return model(**inputs)
        elif isinstance(inputs, collections.abc.Sequence):
            return model(*inputs)
        else:
            return model(inputs)

    @torch.no_grad()
    def test_model_eval(self, on_device=True):
        model = self.framework_model.eval()
        golden = self.get_golden_outputs(model, self.inputs)  # set self.golden_outputs

        if on_device == True:
            model = self.compile_model(model, self.compiler_config)

        outputs = self.run_model(model, self.inputs)
        assert type(outputs) == type(
            golden
        ), "Expecting the type of both calculated and golden to be identical. Whether that be a tensor, list, dictonary, etc."

        passed, err_msg = verify_against_golden(
            self._extract_outputs(golden),
            self._extract_outputs(outputs),
            self.pcc,
            self.required_atol,
            self.relative_atol,
        )
        assert passed, err_msg
        return outputs

    def test_model(self, on_device=True):
        if self.mode == "train":
            return self.test_model_train(on_device)
        elif self.mode == "eval":
            return self.test_model_eval(on_device)
        else:
            raise ValueError(f"Current mode is not supported: {self.mode}")
