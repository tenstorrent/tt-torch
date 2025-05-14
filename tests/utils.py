# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
import requests
import onnx
from onnx.tools import update_model_dims
import onnxruntime
import numpy as np
import collections
import re
from typing import List, Dict, Tuple
from tt_torch.dynamo.backend import backend, BackendOptions
from tt_torch.onnx_compile import compile_onnx
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    prepare_inference_session,
    onnx_output_to_torch,
    torch_input_to_onnx,
)
import json
from onnx import version_converter
from pathlib import Path
from tt_torch.tools.verify import verify_against_golden
from tt_torch.tools.utils import RuntimeIntermediate, OpByOpBackend
import io
import csv
import os


def skip_full_eval_test(
    record_property,
    compiler_config,
    model_name,
    bringup_status,
    reason,
    model_group="generality",
    model_name_filter=None,
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
    Returns:
        bool: True if test was skipped, False otherwise
    """

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
            },
        )
        record_property("group", model_group)

        pytest.skip(reason=reason)
        return True
    return False


class ModelTester:
    def __init__(
        self,
        model_name,
        mode,
        required_pcc=0.99,
        required_atol=None,
        relative_atol=None,
        compiler_config=None,
        assert_pcc=True,
        assert_atol=True,
        record_property_handle=None,
        model_group="generality",
        is_token_output=False,
        model_name_suffix="",
        device=None,
    ):
        if mode not in ["train", "eval"]:
            raise ValueError(f"Current mode is not supported: {mode}")
        self.model_name = model_name
        self.mode = mode
        self.framework_model = self._load_model()
        self.is_token_output = is_token_output
        if is_token_output and not hasattr(self, "tokenizer"):
            raise ValueError(
                "is_token_output is set to True. Please set `self.tokenizer` inside _load_model method."
            )
        self.compiled_model = None
        self.device = device
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
        self.golden_outputs = None
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self.compiler_config = compiler_config
        self.compiler_config.model_name = model_name
        self.compiler_config.model_group = model_group

        self.record_property = record_property_handle
        self.compiler_config.record_property = record_property_handle

        self.record_tag_cache = {}  # Holds for tags to be written out at finalize()

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

        # Postgres jsonb requires lowercased booleans. Pipeline transform from Python->XML->Python->JSON eventually lowercases these
        self.record_tag_cache["is_asserting_pcc"] = self.assert_pcc
        self.record_tag_cache["is_asserting_atol"] = self.assert_atol

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
                return tuple(
                    [t]
                    if not isinstance(t, (tuple, list))
                    else [item for sublist in t for item in flatten(sublist)]
                )

            return flatten(output_object.to_tuple())

        raise NotImplementedError(
            f"Output object type: ({type(output_object)}) is not a torch.Tensor, tuple[torch.Tensor], or list[torch.Tensor], nor does it implement `to_tuple`. Please implement _extract_outputs in the derived class."
        )

    def get_golden_outputs(self, model, inputs):
        if self.golden_outputs is not None:
            return self.golden_outputs

        self.golden_outputs = self.run_model(model, inputs)
        return self.golden_outputs

    def compile_model(self, model, compiler_config):
        # Compile model
        options = BackendOptions()
        options.compiler_config = compiler_config
        options.device = self.device
        model = torch.compile(model, backend=backend, dynamic=False, options=options)
        self.compiled_model = model
        return self.compiled_model

    def run_model(self, model, inputs):
        if isinstance(inputs, collections.abc.Mapping):
            return model(**inputs)
        elif isinstance(inputs, collections.abc.Sequence):
            return model(*inputs)
        else:
            return model(inputs)

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

    def verify_outputs(self, golden, outputs):

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

        pccs, atols, passed_pcc, passed_atol = verify_against_golden(
            golden_tensors,
            output_tensors,
            self.assert_pcc,
            self.assert_atol,
            self.required_pcc,
            self.required_atol,
            self.relative_atol,
        )
        self.record_tag_cache["pccs"] = pccs
        self.record_tag_cache["atols"] = atols
        if passed_pcc and passed_atol:
            self.record_property("achieved_compile_depth", "PASSED")

    def get_framework_model(self):
        model = (
            self.framework_model.eval()
            if hasattr(self.framework_model, "eval")
            else self.framework_model
        )
        return model

    def test_model_eval(self, on_device=True, assert_eval_token_mismatch=True):
        if (
            self.compiler_config.compile_depth == CompileDepth.COMPILE_OP_BY_OP
            or self.compiler_config.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
        ):
            return self._test_model_eval_op_by_op(on_device)
        return self._test_model_eval_base(on_device, assert_eval_token_mismatch)

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

        if self.is_token_output:
            decoded_outputs = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            decoded_golden = self.tokenizer.batch_decode(
                golden, skip_special_tokens=True
            )
            if assert_eval_token_mismatch:
                assert (
                    decoded_outputs == decoded_golden
                ), f'Output mismatch: calculated: "{decoded_outputs} vs golden: "{decoded_golden}"'
        else:
            self.verify_outputs(golden, outputs)
        return outputs

    @torch.inference_mode()
    def _test_model_eval_op_by_op(self, on_device):
        model = self.get_framework_model()

        if on_device == True:
            model = self.compile_model(model, self.compiler_config)

        outputs = self.run_model(model, self.inputs)
        self.record_property("achieved_compile_depth", "EXECUTE")

        return outputs

    def test_model(self, on_device=True, assert_eval_token_mismatch=True):
        if self.mode == "train":
            return self.test_model_train(on_device)
        elif self.mode == "eval":
            return self.test_model_eval(on_device, assert_eval_token_mismatch)
        else:
            raise ValueError(f"Current mode is not supported: {self.mode}")

    def filter_nan_inf_for_record(self, metric_key):
        metric_list = self.record_tag_cache.get(metric_key, [])
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
        self.record_tag_cache[metric_key] = metric_list

    def record_aggregate_model_metric(self, metric_key, default_value=-1):
        # read a metric from the tag cache and write out the average and min values

        metric_list = self.record_tag_cache.get(metric_key, [])
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

        self.record_tag_cache["avg_" + metric_key] = avg_metric
        self.record_tag_cache["min_" + metric_key] = min_metric
        self.record_tag_cache["max_" + metric_key] = max_metric

    def remap_compile_depth(self, compile_depth, min_pcc):
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

        return "PASSED" if min_pcc >= self.required_pcc else "INCORRECT_RESULT"

    def flush_tag_cache_to_record(self):
        # record the tags property at the very end of the test as data may
        # be appended to the cache during the test run

        self.record_property("tags", self.record_tag_cache)

    def finalize(self):
        # to be called at the end of the test

        self.filter_nan_inf_for_record("pccs")
        self.filter_nan_inf_for_record("atols")

        self.record_aggregate_model_metric("pccs")
        self.record_aggregate_model_metric("atols")

        # FE standardization - pack a single PCC & ATOL into the tag record
        # use cached metrics. Guaranteed to be init'd to the default value
        self.record_tag_cache["pcc"] = self.record_tag_cache["min_pccs"]
        self.record_tag_cache["atol"] = self.record_tag_cache["max_atols"]

        # Compile depth is remapped post-execution to FE-standardized format
        # based on actual execution result.
        self.record_tag_cache["bringup_status"] = self.remap_compile_depth(
            self.compiler_config.compile_depth, self.record_tag_cache["min_pccs"]
        )

        self.flush_tag_cache_to_record()

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


class OnnxModelTester(ModelTester):
    def __init__(
        self,
        model_name,
        mode,
        required_pcc=0.99,
        required_atol=None,
        relative_atol=None,
        compiler_config=None,
        assert_pcc=True,
        assert_atol=True,
        record_property_handle=None,
        model_group="generality",
    ):

        super().__init__(
            model_name,
            mode,
            required_pcc,
            required_atol,
            relative_atol,
            compiler_config,
            assert_pcc,
            assert_atol,
            record_property_handle,
            model_group,
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
        self.compiled_model = model
        return self.compiled_model

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

        if isinstance(model, onnx.ModelProto):
            outputs = self.run_model(model, self.numpy_inputs)
        else:
            outputs = self.run_model(model, self.torch_inputs)

        self.verify_outputs(golden, outputs)

        return [
            torch.from_numpy(out) if isinstance(out, np.ndarray) else out
            for out in outputs
        ]


@staticmethod
def get_file(s3_path):
    """
    Please refer to documentation /tt-torch/docs/src/adding_models.md for details on how to add files to S3 bucket and how to use this function to load files locally or from CI.
    """
    rel_dir, file_name = os.path.split(s3_path)
    if (
        "DOCKER_CACHE_ROOT" in os.environ
        and Path(os.environ["DOCKER_CACHE_ROOT"]).exists()
    ):
        cache_dir = (
            Path(os.environ["DOCKER_CACHE_ROOT"])
            / "models/tt-ci-models-private"
            / rel_dir
        )
    elif "LOCAL_LF_CACHE" in os.environ:
        cache_dir = Path(os.environ["LOCAL_LF_CACHE"]) / rel_dir
    else:
        cache_dir = Path.home() / ".cache/lfcache" / rel_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    file_path = cache_dir / file_name
    if not file_path.exists():
        if "DOCKER_CACHE_ROOT" in os.environ:
            raise FileNotFoundError(
                f"File {file_path} is not available, check file path. If path is correct, DOCKER_CACHE_ROOT syncs automatically with S3 bucket every hour so please wait for the next sync."
            )
        else:
            if "IRD_LF_CACHE" not in os.environ:
                raise ValueError(
                    "IRD_LF_CACHE environment variable is not set. Please set it to the address of the IRD LF cache."
                )
            exit_code = os.system(
                f"wget -nH -np -R \"indexg.html*\" -P {cache_dir} {os.environ['IRD_LF_CACHE']}/{s3_path} --connect-timeout=15 --read-timeout=60 --tries=3"
            )
            # Check for wget failure
            if exit_code != 0:
                raise RuntimeError(
                    f"wget failed with exit code {exit_code} when downloading {os.environ['IRD_LF_CACHE']}/{s3_path}"
                )

            # Ensure file_path exists after wget command
            if not file_path.exists():
                raise RuntimeError(
                    f"Download appears to have failed: File {file_name} not found in {cache_dir} after wget command"
                )

    return file_path
