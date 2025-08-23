# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
import requests
import onnx
import tt_torch
from transformers.cache_utils import DynamicCache, _flatten_dynamic_cache
from onnx.tools import update_model_dims
import gc
import onnxruntime
import numpy as np
import collections
from tt_torch.dynamo.backend import BackendOptions
from tt_torch.onnx_compile import compile_onnx
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    prepare_inference_session,
    onnx_output_to_torch,
    torch_input_to_onnx,
    with_torch_dynamo_cleanup,
)
import warnings
from onnx import version_converter
from tt_torch.tools.verify import verify_against_golden
from tt_torch.tools.utils import RuntimeIntermediate, OpByOpBackend
from tt_torch.tools.device_manager import DeviceManager
import io
import csv
import tt_mlir

# Torch-XLA imports
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


def create_device_mesh(mesh_shape, mesh_names):
    assert len(mesh_shape) == len(
        mesh_names
    ), "Mesh shape and names must match in length"
    num_devices = xr.global_runtime_device_count()
    assert (
        np.prod(mesh_shape) == num_devices
    ), "Mesh shape must match the number of devices"
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, mesh_names)
    return mesh


def skip_full_eval_test(
    record_property,
    compiler_config,
    model_name,
    bringup_status,
    reason,
    model_group="generality",
    model_name_filter=None,
    forge_models_test=False,
):
    """
    Helper function to skip a test when frontend has issues and record properties.
    Only skips the test if compile depth is EXECUTE.

    Args:
        record_property: The record_property handle from pytest
        compiler_config: Compiler config to check the compile depth
        model_name: The name of the model being tested
        bringup_status: The bringup status of the test (FAILED_FE_COMPILATION, FAILED_TTMLIR_COMPILATION, FAILED_RUNTIME, INCORRECT_RESULT, PASSED)
        reason: The reason for skipping the test
        model_group: The model group (default: "generality")
        model_name_filter: Either a string or a list of strings. If provided, the test will only be skipped if model_name matches exactly (string) or is in the list (list of strings)
        forge_models_test: Whether the test is a tt-forge-models model test run via test_models.py.
    Returns:
        bool: True if test was skipped, False otherwise
    """
    # Make sure model_group is string for json serializing. It may come from ModelGroup StrEnum.
    model_group = str(model_group)

    # If there is a model name filter applied, and the model name passed herein does not match,
    #   then run the test as normal. Useful for parameterized tests.
    if model_name_filter is not None:
        if isinstance(model_name_filter, list):
            # If it's a list, only skip if the model_name is in the list
            if model_name not in model_name_filter:
                return False
        elif model_name_filter != model_name:
            # If it's a string, only skip if the model_name matches exactly
            return False

    if compiler_config.compile_depth == CompileDepth.EXECUTE:
        record_property(
            "tags",
            {
                "bringup_status": bringup_status,
                "model_name": model_name,
                "forge_models_test": forge_models_test,
            },
        )
        record_property("group", model_group)

        pytest.skip(reason=reason)
        return True
    return False


def clear_dynamo_cache():
    # taken from/ inspired by: https://github.com/pytorch/pytorch/issues/107444
    import torch._dynamo as dynamo

    dynamo.reset()  # clear cache
    gc.collect()


class ModelTester:
    def __init__(
        self,
        model_name,
        mode,
        loader=None,
        model_info=None,
        required_pcc=0.99,
        required_atol=None,
        relative_atol=None,
        compiler_config=None,
        assert_pcc=True,
        assert_atol=True,
        run_generate=False,
        record_property_handle=None,
        model_group="generality",
        is_token_output=False,
        model_name_suffix="",
        devices=None,
        data_parallel_mode=False,
        backend="tt-experimental",
        forge_models_test=False,
    ):
        """
        Initializes the ModelTester.
        Args:
            model_name (str): Name of the model.
            mode (str): "train" or "eval" mode.
            loader (ModelLoader, optional): TT-Forge-Models Loader for the model. Defaults to None.
            model_info(ModelInfo, optional): TT-Forge-Models ModelInfo object for the model. Defaults to None.
                                             When provided is used to extract model_name, model_group, etc.
            required_pcc (float, optional): Required Pearson Correlation Coefficient for verification. Defaults to 0.99.
            required_atol (float, optional): Required absolute tolerance for verification. Defaults to None.
            relative_atol (float, optional): Required relative absolute tolerance for verification. Defaults to None.
            compiler_config (CompilerConfig, optional): Configuration for the compiler. Defaults to None.
            assert_pcc (bool, optional): Whether to assert PCC during verification. Defaults to True.
            assert_atol (bool, optional): Whether to assert ATOL during verification. Defaults to True.
            run_generate (bool, optional): If True, the model's `generate` method will be called for inference.
                                            This is typically used for generative models. Defaults to False.
            record_property_handle (pytest.fixture, optional): Pytest fixture to record properties. Defaults to None.
            model_group (str, optional): Group the model belongs to. Defaults to "generality".
            is_token_output (bool, optional): Flag indicating if the model output is in token form, requiring decoding. Defaults to False.
            model_name_suffix (str, optional): Suffix to append to the model name for recording. Defaults to "".
            devices (List[torch.device] or torch.device, optional): A single `torch.device` or a list of `torch.device`
                                                                    objects to use for compilation and execution. If
                                                                    `data_parallel_mode` is True and `devices` is None,
                                                                    all available devices will be acquired. Defaults to None.
            data_parallel_mode (bool, optional): If True, the model will be compiled and run in a data-parallel fashion
                                                    across the specified `devices`. This mode does not support op-by-op
                                                    compilation or execution. Defaults to False.
            forge_models_test (bool, optional): Whether the test is a tt-forge-models model test run via test_models.py. Defaults to False.
        """
        if mode not in ["train", "eval"]:
            raise ValueError(f"Current mode is not supported: {mode}")

        # ModelInfo object is what we eventually want to move towards using always.
        # If it is provided, use it to extract important fields and override.
        self.model_info = model_info
        if self.model_info is not None:
            model_name = self.model_info.name
            model_group = self.model_info.group

        self.model_name = model_name
        self.loader = loader
        self.mode = mode

        # Make sure model_group is string for json serializing. It may come from ModelGroup StrEnum.
        model_group = str(model_group)

        self.data_parallel_mode = data_parallel_mode
        if int(os.environ.get("TT_TORCH_FORCE_EXPERIMENTAL_BACKEND", False)):
            self.backend = "tt-experimental"
        elif int(os.environ.get("TT_TORCH_FORCE_LEGACY_BACKEND", False)):
            self.backend = "tt"
        else:
            self.backend = backend

        # FIXME - https://github.com/tenstorrent/tt-torch/issues/1105
        # AssertionError: Data parallel mode is not supported with XLA currently
        if self.backend == "tt-experimental" and self.data_parallel_mode:
            print(
                "Data parallel mode is not supported with XLA currently - reverting to legacy"
            )
            self.backend = "tt"

        self.framework_model = self._load_model()
        self.is_token_output = is_token_output
        if is_token_output and not hasattr(self, "tokenizer"):
            raise ValueError(
                "is_token_output is set to True. Please set `self.tokenizer` inside _load_model method."
            )
        self.compiled_models = []
        self.devices = devices
        self.inputs = self._load_inputs()

        self.required_pcc = required_pcc
        self.assert_pcc = assert_pcc
        self.assert_atol = assert_atol

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
        self.run_generate = run_generate  # a model can be generative or discriminative, if `run_generate=True` you invoke `model.generate(**inputs)
        self.golden_outputs = None
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self.compiler_config = compiler_config
        self.compiler_config.model_name = model_name
        self.compiler_config.model_group = model_group

        self.record_property = record_property_handle
        self.compiler_config.record_property = record_property_handle

        # setting consteval parameters to false for the nightly (issue #1182)
        if self.compiler_config is not None:
            self.compiler_config.consteval_parameters = False
        
        self.parent_device = None
        if self.data_parallel_mode:
            assert self.compiler_config.compile_depth not in (
                CompileDepth.COMPILE_OP_BY_OP,
                CompileDepth.EXECUTE_OP_BY_OP,
            ), "Data parallel mode does not support op-by-op compilation or execution."
            if self.devices is None:
                # If user doesn't provide any devices, acquire all devices on board
                (
                    self.parent_device,
                    self.devices,
                ) = DeviceManager.acquire_available_devices()

        self.record_tag_cache = {}  # Holds for tags to be written out at finalize()
        self.verification_failure_msg = (
            None  # Store failure message for deferred assertion
        )

        self.record_tag_cache["backend"] = self.backend
        self.record_property("model_name", model_name + model_name_suffix)
        self.record_property("frontend", "tt-torch")
        self.record_property("owner", "tt-torch")
        self.record_property("group", model_group)

        self.record_tag_cache["model_name"] = model_name + model_name_suffix
        self.record_tag_cache["frontend"] = "tt-torch"

        print("[MODEL NAME]", model_name + model_name_suffix)

        self.record_tag_cache["required_pcc"] = self.required_pcc

        # Avoid introducing conditional logic in DB to handle separate cases
        self.record_tag_cache["required_atol"] = (
            self.required_atol if self.required_atol is not None else self.relative_atol
        )

        self.record_tag_cache["is_asserting_pcc"] = self.assert_pcc
        self.record_tag_cache["is_asserting_atol"] = self.assert_atol

        self.record_tag_cache["parallelism"] = self.get_parallelism()
        self.record_tag_cache["forge_models_test"] = forge_models_test

        # configs should be set at test start, so they can be flushed immediately
        self.record_property(
            "config",
            {"compiler_config": compiler_config.to_dict()},
        )

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
        elif isinstance(output_object, (int, float)):
            return (torch.tensor(output_object),)
        elif isinstance(output_object, str):
            return (output_object,)
        elif isinstance(output_object, (tuple, list)):

            def flatten_tensor_lists(obj):
                flattened = []
                for item in obj:
                    if isinstance(item, torch.Tensor):
                        flattened.append(item)
                    elif isinstance(item, (np.ndarray)):
                        flattened.append(torch.from_numpy(item))
                    elif np.isscalar(item):
                        flattened.append(torch.tensor(item))
                    elif isinstance(item, (tuple, list)):
                        flattened.extend(flatten_tensor_lists(item))
                    else:
                        raise NotImplementedError(
                            f"Item type: ({type(item)}) is not a torch.Tensor or list/tuple of torch.Tensors"
                        )
                return flattened

            try:
                flattened_tensors = flatten_tensor_lists(output_object)
                return tuple(flattened_tensors)
            except NotImplementedError as e:
                raise e
        elif hasattr(output_object, "to_tuple"):

            def flatten(t):
                if isinstance(t, DynamicCache):
                    # `DynamicCache` is most usually returned for generative models, regardless of whether `model(**inputs)` or `model.generate(**inputs)` is called.
                    # `_flatten_dynamic_cache` returns a tuple where the first element is a list of tensors and the second element is a list of `['key_cache', 'value_cache']`
                    # The first element is enough, so we only use that. It is a list so we need to `flatten` it.
                    return flatten(_flatten_dynamic_cache(t)[0])
                elif not isinstance(t, (tuple, list)):
                    return (t,)
                else:
                    # Flatten nested lists/tuples
                    flattened = []
                    for item in t:
                        flattened.extend(flatten(item))
                    return tuple(flattened)

            return flatten(output_object.to_tuple())

        raise NotImplementedError(
            f"Output object type: ({type(output_object)}) is not a torch.Tensor, tuple[torch.Tensor], or list[torch.Tensor], nor does it implement to_tuple. Please implement _extract_outputs in the derived class."
        )

    def get_golden_outputs(self, model, inputs):
        if self.golden_outputs is not None:
            print("Reusing cached golden outputs instead of rerunning the model.")
            return self.golden_outputs

        self.golden_outputs = self.run_model(model, inputs)
        return self.golden_outputs

    def compile_models_for_data_parallel(self, model, compiler_config):
        compiled_models = []
        for device in self.devices:
            compiled_models.append(
                self.compile_model(model, compiler_config, True, device)
            )
        self.compiled_models = compiled_models
        return compiled_models

    def compile_model(
        self, model, compiler_config, data_parallel_mode=False, device_override=None
    ):

        clear_dynamo_cache()
        device = None
        if self.devices is not None and not data_parallel_mode:
            assert (
                isinstance(self.devices, list) and len(self.devices) == 1
            ), "Only a single device may be provided when data_parallel_mode = False"
            device = self.devices[0]
        if device_override:
            device = device_override
        options = BackendOptions()
        options.compiler_config = compiler_config
        options.devices = [device]
        options.async_mode = data_parallel_mode
        # compile forward pass for generative models, the model itself for discriminative
        if self.run_generate:
            model.forward = torch.compile(
                model.forward, backend=self.backend, dynamic=False, options=options
            )
        else:
            model = torch.compile(
                model, backend=self.backend, dynamic=False, options=options
            )
        self.compiled_models.append(model)
        return model

    def run_model(self, model, inputs):
        # call function is determined based on generative vs discriminative
        call_fn = model.generate if self.run_generate else model
        if isinstance(inputs, collections.abc.Mapping):
            return call_fn(**inputs)
        elif isinstance(inputs, collections.abc.Sequence):
            return call_fn(*inputs)
        else:
            return call_fn(inputs)

    def append_fake_loss_function(self, outputs):
        # Using `torch.mean` as the loss function for testing purposes.
        #
        # Loss functions typically produce a scalar loss, and `torch.mean`
        # is one valid option in this category. While it may not be the best
        # choice for training effective models, it simplifies our testing process.
        #
        # Since our goal is to verify gradient computation rather than to
        # train a high-performing model, applying `torch.mean` uniformly
        # across all models under test eases the testing procedure.
        if str(type(outputs)) in [
            "<class 'torch.Tensor'>",
            "<class 'core.TorchTensor'>",
        ]:
            return torch.mean(outputs)
        else:
            raise ValueError(
                f"append_fake_loss_function: Current outputs type is not supported: {type(outputs)}"
            )

    def set_inputs_train(self, inputs):
        if type(inputs) == torch.Tensor:
            # Setting input tensor's `requires_grad` attribute to true.
            #
            # This allows us to use the gradient of the input as the golden result for the training process.
            # For further details, refer to the file `conftest.py` regarding the rationale behind.
            inputs.requires_grad_(True)
        else:
            raise ValueError(
                f"set_inputs_train: Current inputs type is not supported: {type(inputs)}"
            )
        return inputs

    def get_results_train(self, model, inputs, outputs):
        # Why `inputs.requires_grad`?
        #
        # Backward pass computes gradients for all trainable weight tensors.
        # However, verifying every gradient can be costly, especially for
        # large models with many parameters.
        #
        # Instead of checking each gradient, we check the gradient
        # of the "model input" tensor only. Computing the input gradient
        # serves as an indicator for the health of all other gradients.
        # Based on the "chain rule", the input gradient depends on all other
        # gradients, so any incorrect gradient computation should reflect here.
        if type(inputs) == torch.Tensor:
            results = inputs.grad
        elif type(inputs) == dict:
            results = {k: v.grad for k, v in inputs.items()}
        else:
            raise ValueError(
                f"get_results_train: Current inputs type is not supported: {type(inputs)}"
            )
        return results

    def test_model_train(self, on_device=True):
        # Fixing the random seed for reproducibility to ease debugging.
        #
        # Training processes involve more randomness compared to evaluation,
        # such as random initialization of weights.
        # Setting a fixed random seed is crucial for consistent testing
        # and debugging during the training process.
        torch.manual_seed(99)
        model = self.framework_model.train()
        inputs = self.set_inputs_train(self.inputs)
        if on_device == True:
            model = self.compile_model(model, self.compiler_config)
        outputs = self.run_model(model, inputs)
        loss = self.append_fake_loss_function(outputs)
        loss.backward()
        # Again, use the gradient of the input (`test_input.grad`) as the golden result for the training process.
        results = self.get_results_train(model, inputs, outputs)
        return results

    def verify_outputs(self, golden, outputs, defer_assertions=False):

        # Only do golden check if running EXECUTE. Limited value comparing in other situations.
        if self.compiler_config.compile_depth != CompileDepth.EXECUTE:
            print(f"Skipping golden check for {self.compiler_config.compile_depth}")
            return

        assert type(outputs) == type(
            golden
        ), "Expecting the type of both calculated and golden to be identical. Whether that be a tensor, list, dictonary, etc."

        golden_tensors, output_tensors = (), ()

        if isinstance(golden, (tuple, list)):
            for golden_item, output_item in zip(golden, outputs):
                assert type(golden_item) == type(
                    output_item
                ), "Expecting the type of each item in outputs and golden to be identical."
                if isinstance(golden_item, dict):
                    # Verify the keys are the same and extract outputs from dict values
                    sorted_golden = sorted(golden_item.items())
                    sorted_outputs = sorted(output_item.items())
                    for (g_k, g_v), (o_k, o_v) in zip(sorted_golden, sorted_outputs):
                        assert g_k == o_k, f"Keys do not match: {g_k} vs {o_k}"
                        golden_tensors += self._extract_outputs(g_v)
                        output_tensors += self._extract_outputs(o_v)
                else:
                    golden_tensors += self._extract_outputs(golden_item)
                    output_tensors += self._extract_outputs(output_item)
        else:
            golden_tensors = self._extract_outputs(golden)
            output_tensors = self._extract_outputs(outputs)

        # When defer_assertions=True, disable assertions in verify_against_golden
        #   This is because an assertion hit in verify_against_golden will cause
        #   the test to early exit before the record_tag_cache is flushed to xml
        #   and no PCC or metadata will be reported
        assert_pcc = self.assert_pcc if not defer_assertions else False
        assert_atol = self.assert_atol if not defer_assertions else False

        pccs, atols, passed_pcc, passed_atol, atol_thresholds = verify_against_golden(
            golden_tensors,
            output_tensors,
            assert_pcc,
            assert_atol,
            self.required_pcc,
            self.required_atol,
            self.relative_atol,
        )
        self.record_tag_cache["pccs"] = pccs
        self.record_tag_cache["atols"] = atols

        # Store verification failure for deferred assertion
        if defer_assertions and (
            (self.assert_pcc and not passed_pcc)
            or (self.assert_atol and not passed_atol)
        ):

            required_or_relative_atol = (
                (self.required_atol, "(ATOL)")
                if self.required_atol is not None
                else (self.relative_atol, "(RTOL)")
            )
            err_parts = []
            if self.assert_pcc and not passed_pcc:
                err_parts.append(
                    f"PCC check failed. Required: {self.required_pcc}, Got lowest pcc {min(pccs) if pccs else 'N/A'}\nDetail:\n\t"
                    + "\n\t".join(
                        [
                            f"output {i}:{pcc} [req > {self.required_pcc}]"
                            for i, pcc in enumerate(pccs)
                            if pcc < self.required_pcc
                        ]
                    )
                )
            if self.assert_atol and not passed_atol:
                err_parts.append(
                    f"ATOL check failed. Required: {required_or_relative_atol}, Got highest atol {max(atols) if atols else 'N/A'}\nDetail:\n\t"
                    + "\n\t".join(
                        [
                            f"output {i}:{atol} [req < {atol_thresholds[i]}]"
                            for i, atol in enumerate(atols)
                            if atol > atol_thresholds[i]
                        ]
                    )
                )
            self.verification_failure_msg = ";\n".join(err_parts)

        if passed_pcc and passed_atol:
            self.record_property("achieved_compile_depth", "PASSED")

    def get_framework_model(self):
        model = (
            self.framework_model.eval()
            if hasattr(self.framework_model, "eval")
            else self.framework_model
        )
        if hasattr(model, "can_generate") and model.can_generate():
            if not self.run_generate:
                print(
                    "Warning: Model is generative but `run_generate=False`. This will run `model(**inputs)` instead of `model.generate(**inputs)`"
                )
        else:
            assert (
                not self.run_generate
            ), "Model is not generative but `run_generate=True` Please disable `run_generate`"

        return model

    def test_model_eval(self, on_device=True, assert_eval_token_mismatch=True):
        if (
            self.compiler_config.compile_depth == CompileDepth.COMPILE_OP_BY_OP
            or self.compiler_config.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
        ):
            return self._test_model_eval_op_by_op(on_device)
        if self.data_parallel_mode:
            assert (
                self.backend == "tt"
            ), "Data parallel mode is not supported with XLA currently"
            outputs = self._test_model_eval_data_parallel(assert_eval_token_mismatch)
            assert len(outputs) == len(self.devices), "Num outputs != num devices"
            return outputs
        return self._test_model_eval_base(on_device, assert_eval_token_mismatch)

    def _verify_full_execution_output(
        self,
        device_output,
        golden_output,
        assert_eval_token_mismatch,
        defer_assertions=False,
    ):
        """
        This function verifies a single device's output tensors against the golden tensors
        (found by running the model on the CPU). This should only be used during full
        model execution, and not in op-by-op mode.
        """
        if self.is_token_output:
            decoded_outputs = self.tokenizer.batch_decode(
                device_output, skip_special_tokens=True
            )
            decoded_golden = self.tokenizer.batch_decode(
                golden_output, skip_special_tokens=True
            )
            if assert_eval_token_mismatch and not defer_assertions:
                assert (
                    decoded_outputs == decoded_golden
                ), f'Output mismatch: calculated: "{decoded_outputs} vs golden: "{decoded_golden}"'
            elif (
                defer_assertions
                and assert_eval_token_mismatch
                and decoded_outputs != decoded_golden
            ):
                self.verification_failure_msg = f'Token output mismatch: calculated: "{decoded_outputs}" vs golden: "{decoded_golden}"'
        else:
            self.verify_outputs(golden_output, device_output, defer_assertions)

    @torch.inference_mode()
    def _test_model_eval_data_parallel(self, assert_eval_token_mismatch):
        model = self.get_framework_model()
        golden = self.get_golden_outputs(model, self.inputs)

        compiled_models = self.compile_models_for_data_parallel(
            model, self.compiler_config
        )

        rt_tensors = []
        for compiled in compiled_models:
            rt_tensor = self.run_model(compiled, self.inputs)
            rt_tensors.append(rt_tensor)

        final_outputs = []
        for rt_tensor in rt_tensors:
            torch_tensors = tt_mlir.to_host(rt_tensor)
            if isinstance(torch_tensors, list):
                if len(torch_tensors) == 1:
                    final_outputs.extend(torch_tensors)
                else:
                    final_outputs.append(tuple(torch_tensors))
            else:
                final_outputs.append(torch_tensors)

        self.record_property("achieved_compile_depth", "EXECUTE")
        if self.compiler_config._enable_intermediate_verification:
            warnings.warn(
                "Runtime intermediate verification is not supported in data parallel mode. Ignoring this."
            )
        try:
            for outputs in final_outputs:
                self._verify_full_execution_output(
                    outputs, golden, assert_eval_token_mismatch, defer_assertions=True
                )
        finally:
            if self.parent_device is not None:
                # The model tester object is managing the devices, release all devices.
                DeviceManager.release_parent_device(
                    self.parent_device, cleanup_sub_devices=True
                )
        return final_outputs

    @torch.inference_mode()
    def _test_model_eval_base(self, on_device, assert_eval_token_mismatch):
        model = self.get_framework_model()
        golden = self.get_golden_outputs(model, self.inputs)

        if on_device == True:
            model = self.compile_model(model, self.compiler_config)

        outputs = self.run_model(model, self.inputs)
        self.record_property("achieved_compile_depth", "EXECUTE")

        if self.compiler_config._enable_intermediate_verification:
            self.verify_intermediates_after_execution()

        self._verify_full_execution_output(
            outputs, golden, assert_eval_token_mismatch, defer_assertions=True
        )
        return outputs

    @torch.inference_mode()
    def _test_model_eval_op_by_op(self, on_device):
        model = self.get_framework_model()

        if on_device == True:
            model = self.compile_model(model, self.compiler_config)

        outputs = self.run_model(model, self.inputs)
        self.record_property("achieved_compile_depth", "EXECUTE")

        return outputs

    @with_torch_dynamo_cleanup
    def test_model(self, on_device=True, assert_eval_token_mismatch=True):
        if self.mode == "train":
            return self.test_model_train(on_device)
        elif self.mode == "eval":
            return self.test_model_eval(on_device, assert_eval_token_mismatch)
        else:
            raise ValueError(f"Current mode is not supported: {self.mode}")

    @staticmethod
    def filter_nan_inf_for_record(record_tag_cache, metric_key):
        # Filters record_tag_cache inplace to strip out nan/inf which break serialization and reporting
        metric_list = record_tag_cache.get(metric_key, [])
        if metric_list:
            metric_list = [
                -1
                if (isinstance(x, float) and x != x)  # NaN case
                else -1
                if (isinstance(x, float) and x == float("inf"))  # +inf
                else -1
                if (isinstance(x, float) and x == -float("inf"))  # -inf
                else 2**32 * (1 if x > 0 else -1)
                if (
                    isinstance(x, (int, float)) and abs(x) > 2**32
                )  # clamp to ±2**32; superset bug
                else x
                for x in metric_list
                if isinstance(x, (int, float)) or isinstance(x, str)
            ]
        record_tag_cache[metric_key] = metric_list

    @staticmethod
    def record_aggregate_model_metric(record_tag_cache, metric_key, default_value=-1):
        # read a metric from the tag cache and write out the average and min values

        metric_list = record_tag_cache.get(metric_key, [])
        avg_metric = default_value
        min_metric = default_value
        max_metric = default_value
        filtered_metrics = [
            x for x in metric_list if x is not None
        ]  # remove None values

        if filtered_metrics:  # check null or empty list
            filtered_metrics = [
                x for x in filtered_metrics if isinstance(x, (int, float))
            ]
            avg_metric = sum(filtered_metrics) / len(filtered_metrics)
            min_metric = min(filtered_metrics)
            max_metric = max(filtered_metrics)

        record_tag_cache["avg_" + metric_key] = avg_metric
        record_tag_cache["min_" + metric_key] = min_metric
        record_tag_cache["max_" + metric_key] = max_metric

    @staticmethod
    def remap_compile_depth(compile_depth, min_pcc, required_pcc):
        compile_depth_translation_table = {
            CompileDepth.TORCH_FX: "FAILED_TTMLIR_COMPILATION",
            CompileDepth.STABLEHLO: "FAILED_RUNTIME",
            CompileDepth.TTNN_IR: "FAILED_RUNTIME",
            CompileDepth.COMPILE_OP_BY_OP: "INCORRECT_RESULT",
            CompileDepth.EXECUTE_OP_BY_OP: "INCORRECT_RESULT",
            CompileDepth.EXECUTE: "INCORRECT_RESULT",  # ambiguous between incorrect / passed
        }

        if compile_depth is not CompileDepth.EXECUTE:
            return compile_depth_translation_table[compile_depth]

        return "PASSED" if min_pcc >= required_pcc else "INCORRECT_RESULT"

    def finalize(self):
        # to be called at the end of the test

        ModelTester.filter_nan_inf_for_record(self.record_tag_cache, "pccs")
        ModelTester.filter_nan_inf_for_record(self.record_tag_cache, "atols")

        ModelTester.record_aggregate_model_metric(self.record_tag_cache, "pccs")
        ModelTester.record_aggregate_model_metric(self.record_tag_cache, "atols")

        # FE standardization - pack a single PCC & ATOL into the tag record
        # use cached metrics. Guaranteed to be init'd to the default value
        self.record_tag_cache["pcc"] = self.record_tag_cache["min_pccs"]
        self.record_tag_cache["atol"] = self.record_tag_cache["max_atols"]

        # Compile depth is remapped post-execution to FE-standardized format
        # based on actual execution result.
        self.record_tag_cache["bringup_status"] = ModelTester.remap_compile_depth(
            self.compiler_config.compile_depth,
            self.record_tag_cache["min_pccs"],
            self.required_pcc,
        )
        self.record_property("tags", self.record_tag_cache)

        # Assert any deferred verification failures after all reporting is complete
        # This allows pytest to write junitxml during teardown even if the test fails
        if self.verification_failure_msg:
            assert False, self.verification_failure_msg

    def verify_intermediates_after_execution(self):
        # Prepare CSV output
        output = io.StringIO()
        csv_writer = csv.writer(output)
        op_pcc_fail_threshold = float(os.environ.get("RTI_PCC_FAIL_THRESH", 0.99))
        last_row = None
        first_failing_row = None

        header = [
            "NodeName",
            "PCC",
            "ATOL",
            "ErrorMessage",
            "FlattenedPCC",
            "FlattenedATOL",
            "FlattenedErrorMessage",
        ]
        # Write the header
        csv_writer.writerow(header)

        # Helper function to unpack single-element lists/tuples
        def unpack(value):
            if isinstance(value, (list, tuple)) and len(value) == 1:
                return value[0]
            return value

        # Write each intermediate's metrics as CSV; sanitize out commas
        for _, intermediate in self.compiler_config.runtime_intermediate_cache.items():
            intermediate.calculate_metrics()
            row = [
                intermediate.node.name,
                unpack(intermediate.pcc),
                unpack(intermediate.atol),
                intermediate.error_message.replace(",", ";")
                if intermediate.error_message
                else "",
                unpack(intermediate.flattened_pcc),
                unpack(intermediate.flattened_atol),
                intermediate.flattened_error_message.replace(",", ";")
                if intermediate.flattened_error_message
                else "",
            ]
            csv_writer.writerow(row)

            # Update the last row
            last_row = row

            # Check if this row has a numeric PCC less than the threshold
            pcc_value = unpack(intermediate.pcc)
            if (
                isinstance(pcc_value, (int, float))
                and pcc_value < op_pcc_fail_threshold
            ):
                if first_failing_row is None:
                    first_failing_row = row

        print("[Start Intermediate Verification Report]")
        print(output.getvalue())
        print("[End Intermediate Verification Report]")

        # Print Summary Info
        print("[Intermediate Verification Summary]")
        print(",".join(header))
        print("Final Row:", ",".join([str(el) for el in last_row]))
        print(f"First Failing Op with PCC < {op_pcc_fail_threshold}", end=": ")
        if first_failing_row:
            print(",".join([str(el) for el in first_failing_row]))
        else:
            print("No failing operations found.")
        print("[End Intermediate Verification Summary]")

    @staticmethod
    def print_outputs(results, data_parallel_mode, print_fn):
        if data_parallel_mode:
            assert isinstance(
                results, list
            ), "Results should be a list in data parallel mode"
            for i, result in enumerate(results):
                print(f"Results for device {i}:")
                print_fn(result)
        else:
            print_fn(results)

    @staticmethod
    def _get_parallelism(data_parallel_mode, automatic_parallelization):
        parallelism = "single_device"

        assert not (
            data_parallel_mode and automatic_parallelization
        ), "Cannot use runtime data parallel and automatic data parallel settings at the same time."

        if data_parallel_mode:
            parallelism = "runtime_data_parallel"

        if automatic_parallelization:
            parallelism = "data_parallel"

        return parallelism

    def get_parallelism(self):
        return ModelTester._get_parallelism(
            self.data_parallel_mode, self.compiler_config.automatic_parallelization
        )

    @staticmethod
    def GenerateCustomTestReport(
        record_property,
        model_name,
        compiler_config,
        test_pcc,
        test_atol,
        model_group="generality",
        required_pcc=0.99,
        required_atol=None,
        relative_atol=None,
        assert_pcc=True,
        assert_atol=True,
        data_parallel_mode=False,
        automatic_parallelization=False,
    ):
        """
        Test reporting is tightly coupled to the ModelTester, under the assumption that almost all
        tests go through the ModelTester. For those tests that do not go through the ModelTester,
        this function can be used to generate a custom test report.

        It imitates the mechanics of ModelTester.finalize() and ModelTester ctor
        """
        record_tag_cache = {}  # Holds for tags to be written out at finalize()

        record_property("model_name", model_name)
        record_property("frontend", "tt-torch")
        record_property("owner", "tt-torch")
        record_property("group", model_group)

        record_tag_cache[
            "note"
        ] = "Using custom test report generation path, not tt-torch ModelTester."
        record_tag_cache["required_pcc"] = required_pcc

        record_tag_cache["required_atol"] = (
            required_atol if required_atol is not None else relative_atol
        )

        record_tag_cache["is_asserting_pcc"] = assert_pcc
        record_tag_cache["is_asserting_atol"] = assert_atol

        record_tag_cache["parallelism"] = ModelTester._get_parallelism(
            data_parallel_mode, automatic_parallelization
        )

        # configs should be set at test start, so they can be flushed immediately
        record_property(
            "config",
            {"compiler_config": compiler_config.to_dict()},
        )

        ModelTester.filter_nan_inf_for_record(record_tag_cache, "pccs")
        ModelTester.filter_nan_inf_for_record(record_tag_cache, "atols")

        ModelTester.record_aggregate_model_metric(record_tag_cache, "pccs")
        ModelTester.record_aggregate_model_metric(record_tag_cache, "atols")

        # FE standardization - pack a single PCC & ATOL into the tag record
        # use cached metrics. Guaranteed to be init'd to the default value
        record_tag_cache["pcc"] = record_tag_cache["min_pccs"]
        record_tag_cache["atol"] = record_tag_cache["max_atols"]

        # Compile depth is remapped post-execution to FE-standardized format
        # based on actual execution result.
        record_tag_cache["bringup_status"] = ModelTester.remap_compile_depth(
            compiler_config.compile_depth, record_tag_cache["min_pccs"], required_pcc
        )

        record_property("tags", record_tag_cache)


# TODO - hshahTT: Add support for data parallel mode for onnx models
class OnnxModelTester(ModelTester):
    def __init__(
        self,
        model_name,
        mode,
        loader=None,
        model_info=None,
        required_pcc=0.99,
        required_atol=None,
        relative_atol=None,
        compiler_config=None,
        assert_pcc=True,
        assert_atol=True,
        run_generate=False,
        record_property_handle=None,
        model_group="generality",
        is_token_output=False,
        model_name_suffix="",
        devices=None,
        data_parallel_mode=False,
    ):

        super().__init__(
            model_name,
            mode,
            loader,
            model_info,
            required_pcc,
            required_atol,
            relative_atol,
            compiler_config,
            assert_pcc,
            assert_atol,
            run_generate,
            record_property_handle,
            model_group,
            is_token_output,
            model_name_suffix,
            devices,
            data_parallel_mode,
        )
        # Hold an onnxruntime session for golden / non-full compile execution
        self.sess = prepare_inference_session(model_proto=self.framework_model)
        self.torch_inputs = self._load_torch_inputs()
        self.numpy_inputs = self._load_numpy_inputs()

    # Pass this function so we can use superclass __init__ without failure
    def _load_inputs(self):
        pass

    def _load_torch_inputs(self):
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def _load_numpy_inputs(self):
        return torch_input_to_onnx(self.sess, self.torch_inputs)

    def compile_model(self, model, compiler_config):
        model = compile_onnx(model, compiler_config)
        self.compiled_models.append(model)
        return model

    def run_model(self, model, inputs):
        if isinstance(model, onnx.ModelProto):
            outputs = self.sess.run(None, inputs)
            return onnx_output_to_torch(outputs)
        elif isinstance(inputs, collections.abc.Mapping):
            return model(**inputs)
        elif isinstance(inputs, collections.abc.Sequence):
            return model(*inputs)
        else:
            return model(inputs)

    def test_model_train(self, on_device=True):
        raise NotImplementedError("TODO: Implement this method")

    def test_model_eval(self, on_device=True, _assert_eval_token_mismatch=False):
        golden = self.run_model(self.framework_model, self.numpy_inputs)

        if on_device == True:
            model = self.compile_model(self.framework_model, self.compiler_config)
        else:
            model = self.framework_model

        if isinstance(model, onnx.ModelProto):
            outputs = self.run_model(model, self.numpy_inputs)
        else:
            outputs = self.run_model(model, self.torch_inputs)

        self.verify_outputs(golden, outputs)

        return [
            torch.from_numpy(out) if isinstance(out, np.ndarray) else out
            for out in outputs
        ]
