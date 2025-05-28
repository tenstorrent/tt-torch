# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import onnx
import onnxruntime as ort
import tt_mlir
import time
import pickle
import faulthandler
import sys
import re
import tempfile
import os
import multiprocessing as mp
from tt_torch.tools.utils import (
    CompileDepth,
    IOType,
    CompilerConfig,
    OpCompilationStatus,
)
from tt_torch.tools.utils import (
    run_model_proto,
    onnx_output_to_torch,
    torch_input_to_onnx,
    MultiChipGraph,
)
from typing import Union, Optional


def gb_to_bytes(gb):
    return gb * 1024 * 1024 * 1024


def get_tensor_size(tensor):
    """Calculate the memory size of a tensor in bytes."""
    if isinstance(tensor, torch.Tensor):
        return tensor.element_size() * tensor.nelement()
    return 0


def get_inputs_size(inputs):
    """Calculate the total memory size of inputs in bytes."""
    total_size = 0

    if isinstance(inputs, torch.Tensor):
        total_size += get_tensor_size(inputs)
    elif isinstance(inputs, (list, tuple)):
        for item in inputs:
            total_size += get_inputs_size(item)
    elif isinstance(inputs, dict):
        for item in inputs.values():
            total_size += get_inputs_size(item)
    else:
        assert False, f"Unexpected input type: {type(inputs)}"
    return total_size


def compile_process(receiver, sender, ttir_event, ttnn_event, json_event):
    obj = receiver.get()
    faulthandler.disable()
    asm = obj["asm"]
    system_desc_path = obj["system_desc_path"]
    ttir = tt_mlir.compile_stable_hlo_to_ttir(asm)
    sender.put({"ttir": ttir})
    ttir_event.wait()
    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir, system_desc_path)
    sender.put({"binary": binary, "ttnn": ttnn})
    ttnn_event.wait()
    sender.put({"json": tt_mlir.bytestream_to_json(binary)})
    json_event.wait()
    sys.exit(0)


def print_tensor_shapes(inputs, prefix=""):
    """Print the shapes and dtypes of tensor inputs for debugging."""
    if isinstance(inputs, torch.Tensor):
        print(f"{prefix}dtype = {inputs.dtype}, shape = {inputs.shape}")
    elif isinstance(inputs, (list, tuple)):
        print(f"{prefix}List/Tuple of length {len(inputs)}")
        for i, item in enumerate(inputs):
            print_tensor_shapes(item, prefix=f"{prefix}[{i}].")
    elif isinstance(inputs, dict):
        print(f"{prefix}Dict with keys: {list(inputs.keys())}")
        for key, item in inputs.items():
            print_tensor_shapes(item, prefix=f"{prefix}['{key}'].")
    else:
        print(f"{prefix}Not a tensor: {type(inputs)}")


def execute_process(receiver, sender, exec_event):
    while 1:
        obj = receiver.get()
        faulthandler.disable()
        binary = obj["binary"]
        file_name = obj["dump_file"]
        large_input = obj["large_input"]
        inputs = None

        # Load inputs from disk if they're large
        if large_input:
            print("Child process handling large input", flush=True)
            inputs_file_path = obj["inputs_file_path"]
            if inputs_file_path and os.path.exists(inputs_file_path):
                try:
                    with open(inputs_file_path, "rb") as f:
                        inputs = pickle.load(f)
                except Exception as e:
                    print(f"Error loading inputs from disk: {e}")
        else:
            inputs = obj["inputs"]

        file_stderr = open(file_name, "w")
        old_stderr = sys.stderr
        sys.stderr = file_stderr
        old_stdout = sys.stdout
        sys.stdout = file_stderr

        outputs = None
        if inputs is not None:
            outputs = tt_mlir.run_on_default_device(inputs, binary)

        sys.stderr = old_stderr
        sys.stdout = old_stdout
        file_stderr.close()

        sender.put({"outputs": outputs})
        exec_event.wait()

    sys.exit(0)


class Executor:
    def __init__(
        self,
        mcg: Optional[MultiChipGraph] = None,
        compiler_config: Optional[CompilerConfig] = None,
        required_pcc=0.99,
        required_atol=1e-2,
        devices=None,
        async_mode=False,
    ):
        self.mcg = mcg
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self.compiler_config = compiler_config
        self.required_atol = required_atol
        self.required_pcc = required_pcc

        # Dictionary to keep track of the type conversion for unsupported hardware
        # types and use it to convert the input arguments to supported types.
        self.type_conversion = {
            torch.bool: torch.bfloat16,
            torch.int64: torch.int32,
            torch.float64: torch.float32,
        }

        self.binary = {}
        self.preprocessed_graph_constants = {}
        self.devices = devices if devices is not None else []
        self.owned_device_indices = []
        self.async_mode = async_mode
        self._validate_executor()
        self.system_desc_paths = self._create_system_descriptors()

    def _create_system_descriptors(self):
        if self.compiler_config.compile_depth in [
            CompileDepth.TORCH_FX,
            CompileDepth.STABLEHLO,
        ]:
            return []

        if not self.devices:
            descriptor_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=".ttsys"
            ).name
            tt_mlir.create_default_system_desc(descriptor_path)
            return [descriptor_path]

        system_desc_paths = []
        for device_idx, device_from_user in enumerate(self.devices):
            descriptor_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=".ttsys"
            ).name
            descriptor_device = self._get_device(device_idx)

            tt_mlir.create_system_desc(descriptor_device, descriptor_path)

            if device_from_user is None:
                self._cleanup_resources([], device_idx)

            system_desc_paths.append(descriptor_path)
        return system_desc_paths

    def _validate_executor(self):
        if self.compiler_config.compile_depth in (
            CompileDepth.EXECUTE_OP_BY_OP,
            CompileDepth.COMPILE_OP_BY_OP,
        ):
            assert (
                self.async_mode is False
            ), "Op-by-op execution does not support async mode."

    def register_intermediate_callback(self, callback):
        if not tt_mlir.is_runtime_debug_enabled():
            raise RuntimeError(
                "Runtime debug is required to use intermediate callbacks. Please recompile this project with -DTT_RUNTIME_DEBUG=ON."
            )
        tt_mlir.DebugHooks.get_debug_hooks(callback)

    def typecast_inputs(self, inputs):
        new_inputs = ()
        for input in inputs:
            # Handle scalar inputs.
            if not hasattr(input, "dtype"):
                new_inputs = new_inputs + ((input),)
                continue

            # Apply type conversion if required.
            input_type = input.dtype
            if input_type in self.type_conversion.keys():
                new_inputs = new_inputs + (
                    (input.to(dtype=self.type_conversion[input_type])),
                )
                continue

            # No conversion required.
            new_inputs = new_inputs + ((input),)
        return new_inputs

    def _get_device(self, device_idx=0):
        assert (
            len(self.devices) >= device_idx
        ), f"Not enough devices provided: {len(self.devices)} <= {device_idx}"
        assert isinstance(
            self.devices[device_idx], tt_mlir.Device
        ), f"Expecting a tt_mlir.Device, received: {type(self.devices[device_idx])}"
        return self.devices[device_idx]

    def _cache_constants_if_needed(self, preprocessed_constants, device_idx=0):
        if (
            self.compiler_config.cache_preprocessed_constants
            and self.graph_constants is not None
            and self.preprocessed_graph_constants[device_idx] is None
        ):
            self.preprocessed_graph_constants[device_idx] = preprocessed_constants

    def _cleanup_resources(self, preprocessed_activations, device_idx):
        for t in preprocessed_activations:
            tt_mlir.deallocate_tensor(t, force=True)

    def get_inputs(self, *inputs, binary, program_idx, device_idx=0):
        def get_torch_tensors(tensors):
            torch_tensors = []
            indices = []
            for idx, tensor in enumerate(tensors):
                if isinstance(tensor, torch.Tensor):
                    torch_tensors.append(tensor)
                    indices.append(idx)
            return torch_tensors, indices

        def recreate_runtime_tensors(tensors, runtime_tensors, indices):
            tensors = list(tensors)
            for index in indices:
                tensors[index] = runtime_tensors.pop(0)
            return tuple(tensors)

        input_len = len(inputs)
        tensor_start_idx = 0
        if device_idx in self.preprocessed_graph_constants:
            preprocessed_weights = self.preprocessed_graph_constants[device_idx]
            weights_and_activations = preprocessed_weights + inputs
            tensor_start_idx = len(preprocessed_weights)
        elif self.mcg.constant_inputs[device_idx] is not None:
            weights_and_activations = (
                tuple(self.mcg.constant_inputs[device_idx]) + inputs
            )
        else:
            weights_and_activations = inputs

        torch_weights_and_activations, torch_indices = get_torch_tensors(
            weights_and_activations
        )

        if self.compiler_config.typecast_inputs:
            torch_weights_and_activations = self.typecast_inputs(
                torch_weights_and_activations
            )

        runtime_activations_and_weights = tt_mlir.preprocess_inputs(
            self._get_device(device_idx=device_idx),
            torch_weights_and_activations,
            binary,
            program_idx,
            tensor_start_idx,
        )
        runtime_activations_and_weights = recreate_runtime_tensors(
            weights_and_activations, runtime_activations_and_weights, torch_indices
        )
        runtime_weights = runtime_activations_and_weights[:-input_len]
        self.preprocessed_graph_constants[device_idx] = tuple(runtime_weights)
        runtime_activations = runtime_activations_and_weights[-input_len:]

        return runtime_weights, runtime_activations

    def __call__(self, *inputs):
        """
        Execute the model with the given inputs.

        If self.async_mode is True, this function will return the on-device runtime
        tensors.
        If self.async_mode is False, this function will move the runtime tensors to host
        and return them as Torch tensors.
        """
        if self.compiler_config.compile_depth != CompileDepth.EXECUTE:
            assert (
                len(self.mcg.programs) == 1
                and self.mcg.programs[0].graph_module != None
            ), "Cannot run base executor without torch graph"
            return self.mcg.programs[0].graph_module(
                *tuple(self.mcg.constant_inputs[0])
                + tuple(self.mcg.programs[0].buffers())
                + inputs
            )

        assert len(self.mcg.binaries) > 0
        intermediate_results = []
        num_outputs = 0
        graph_inputs = {}
        for device_idx in self.mcg.binaries.keys():
            for output in self.mcg.graph_outputs[device_idx]:
                if output.io_type == IOType.USER:
                    num_outputs += 1
            graph_inputs[device_idx] = [None] * len(self.mcg.graph_inputs[device_idx])
            for input in self.mcg.graph_inputs[device_idx]:
                if input.io_type == IOType.USER:
                    graph_inputs[device_idx][input.consumer_index] = inputs[
                        input.producer_index
                    ]

        final_outputs = [None] * num_outputs
        for device_idx, binary in self.mcg.binaries.items():
            device_inputs = graph_inputs[device_idx]

            binary = tt_mlir.create_binary_from_bytestream(binary)
            program_idx = 0
            preprocessed_weights, preprocessed_activations = self.get_inputs(
                *device_inputs,
                binary=binary,
                device_idx=device_idx,
                program_idx=program_idx,
            )

            device_inputs = list(device_inputs)
            device = self._get_device(device_idx)

            # if any output is intermediate we can run in async, since tt-mlir runtime will eventually block on final outputs
            # TODO: Enable this when device to device movement is supported. In the mean time we fall back to host: #748
            # intermediate_output = any([o.io_type == IOType.INTER_DEVICE for o in self.mcg.graph_outputs[device_idx]])
            intermediate_output = False
            if self.async_mode or intermediate_output:
                outputs = tt_mlir.run_async(
                    device,
                    binary,
                    program_idx,
                    preprocessed_weights + preprocessed_activations,
                )
            else:
                outputs = tt_mlir.run(
                    device,
                    binary,
                    program_idx,
                    (preprocessed_weights + preprocessed_activations),
                )

            for i, output in enumerate(outputs):
                graph_output = self.mcg.graph_outputs[device_idx][i]
                if graph_output.io_type == IOType.INTER_DEVICE:
                    mci = graph_output.linked_input
                    graph_inputs[mci.originating_device][mci.consumer_index] = output
                    intermediate_results.append(output)
                else:
                    final_outputs[graph_output.index] = output

        self._cleanup_resources(preprocessed_activations, device_idx)
        assert all([o is not None for o in final_outputs])
        return final_outputs

    def __del__(self):
        for _, device_weights in self.preprocessed_graph_constants.items():
            for weight in device_weights:
                tt_mlir.deallocate_tensor(weight, force=True)
        for path in self.system_desc_paths:
            try:
                os.remove(path)
            except OSError:
                pass


class OnnxExecutor(Executor):
    def __init__(self, model_proto: onnx.ModelProto, devices):
        super().__init__(devices=devices)
        self.model_proto = model_proto
        self.binary = None
        self.sess = None
        self.compiler_config = CompilerConfig()
        self.preprocessed_graph_constants = {}
        self.owned_device_indices = []
        self.system_desc_paths = self._create_system_descriptors()

    def typecast_inputs(self, inputs):
        raise NotImplementedError("This should not be called on an OnnxExecutor.")

    # Todo: refactor onnx executor to use mcg as well
    def set_binary(self, binary):
        self.binary = binary

    def __call__(self, *inputs):
        assert (
            self.compiler_config.compile_depth == CompileDepth.EXECUTE
            or self.compiler_config.compile_depth == CompileDepth.TTNN_IR
        ), "OnnxExecutor does not support op-by-op flow, please use StablehloExecutor"
        if (
            self.binary is None
            or self.compiler_config.compile_depth == CompileDepth.TTNN_IR
        ):
            # Only want to load the model proto into one inference session
            # since models can be big
            output = run_model_proto(
                sess=self.sess, model_proto=self.model_proto, inputs=inputs
            )
            return onnx_output_to_torch(output)

        return tt_mlir.run_end_to_end(self.devices[0], inputs, self.binary)


class OpByOpExecutor(Executor):

    # Class attributes for identifying each op w/ unique incrementing id
    # across graph breaks, and for running just a specific op.
    global_op_idx = 0
    run_global_op_idx = None
    compiling_time = 0.0
    running_time = 0.0
    golden_time = 0.0

    def __init__(
        self,
        mcg=None,
        compiler_config=None,
        required_pcc=0.99,
        required_atol=1e-2,
        devices=None,
        async_mode=False,
    ):
        if mcg is not None:
            assert len(mcg.programs) == 1

        super().__init__(
            mcg=mcg,
            compiler_config=compiler_config,
            required_pcc=required_pcc,
            required_atol=required_atol,
            devices=devices,
            async_mode=async_mode,
        )

        # Debug mode to run only specific op given global_op_idx
        if OpByOpExecutor.run_global_op_idx is None:
            run_global_op_idx_env = os.getenv("RUN_GLOBAL_OP_IDX")
            OpByOpExecutor.run_global_op_idx = (
                None if run_global_op_idx_env is None else int(run_global_op_idx_env)
            )

        # Opening a device in a new process is very slow as the pcie device needs to be initializes
        # So we keep the process alive and reuse it. If the process dies, the next call will create a new process
        self.execute_process = None
        self.execute_sender = None
        self.execute_receiver = None

        # Create temp file at start of execution of first op and pass the name
        # of temp file to subprocess which will be used to redirect the stderr
        # to capture runtime stack dump.
        self.stderror_redirected = False
        self.file_stderr = None
        self.op_memory_limit = gb_to_bytes(0.5)  # 512MB limit

    # Determine if the current op should be tested based on RUN_GLOBAL_OP_IDX
    def should_test_op(self):
        return (
            OpByOpExecutor.run_global_op_idx is None
            or OpByOpExecutor.global_op_idx == OpByOpExecutor.run_global_op_idx
        )

    def transform_input(self, inp):
        # Convert torch.nn.Parameter to torch.Tensor and convert non-contiguous
        # data to contiguous.
        if isinstance(inp, torch.nn.Parameter):
            if not inp.data.is_contiguous():
                inp.data = inp.data.contiguous()
            return inp.data
        elif isinstance(inp, torch.Tensor):
            if not inp.is_contiguous():
                inp = inp.contiguous()
            return inp

        return None

    def pre_process_inputs(self, *inputs):
        # Remove scalar constants as they're absorbed into the binary
        processed_inputs = []
        for input in inputs:
            # If input is a list, iterate over its elements;
            # otherwise, process it directly
            input_items = input if isinstance(input, list) else [input]

            for inp in input_items:
                transformed_inp = self.transform_input(inp)
                if transformed_inp is not None:
                    processed_inputs.append(transformed_inp)
        supported_inputs = self.typecast_inputs(processed_inputs)
        return supported_inputs

    def get_input_shapes_and_constants(self, *inputs):
        input_shapes_and_constants = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                input_shapes_and_constants.append(inp.shape)
            elif isinstance(inp, (list, tuple)):
                sub = []
                for sub_inp in inp:
                    if isinstance(sub_inp, torch.Tensor):
                        sub.append(sub_inp.shape)
                    else:
                        sub.append(sub_inp)
                input_shapes_and_constants.append(sub)
            elif isinstance(inp, (int, float, bool)):
                input_shapes_and_constants.append(inp)
            elif isinstance(inp, torch.dtype):
                input_shapes_and_constants.append(inp.__str__())
            elif inp is None:
                input_shapes_and_constants.append(None)
            else:
                raise ValueError(f"Unexpected input type: {type(inp)}")
        return input_shapes_and_constants

    def set_runtime_stack_dump(self, error, op):
        if op is None:
            return

        # Handle both implementations of unique_key (method or attribute)
        key = (
            op.unique_key()
            if callable(getattr(op, "unique_key", None))
            else op.unique_key
        )
        self.compiler_config.unique_ops[key].runtime_stack_dump = str(error)

    # Helper function to print markers
    def print_marker(self, msg, idx, num_nodes, op_info, error="", time=0.0):
        print(
            f"{msg:<10} global_op_idx: {OpByOpExecutor.global_op_idx} ({idx}/{num_nodes}): {op_info} | time: {time:.4f} s | {error}",
            flush=True,
        )

    def compile_op(self, node, *inputs, **kwargs):
        # get_stablehlo_graph is a method implemented in inheriting classes
        module, op = self.get_stable_hlo_graph(node, inputs, **kwargs)

        if module is None or op is None:
            return None, None, None

        sender = mp.Queue()
        receiver = mp.Queue()
        ttir_event = mp.Event()
        ttnn_event = mp.Event()
        json_event = mp.Event()
        obj = {
            "asm": module.operation.get_asm(),
            "system_desc_path": self.system_desc_paths[0],
        }
        process = mp.Process(
            target=compile_process,
            args=(sender, receiver, ttir_event, ttnn_event, json_event),
        )
        process.start()
        sender.put(obj)
        start = time.time()
        binary = None
        msg = None
        timeout_exceeded = False
        while True:
            try:
                result = receiver.get_nowait()
                if "ttir" in result:
                    op.compilation_status = OpCompilationStatus.CONVERTED_TO_TTIR
                    op.add_ttir_graph(result["ttir"])
                    ttir_event.set()
                if "binary" in result:
                    binary = result["binary"]
                    op.binary = binary
                    op.add_ttnn_graph(result["ttnn"])
                    ttnn_event.set()
                if "json" in result:
                    op.json = result["json"]
                    json_event.set()
                    op.parse_json()
                    op.compilation_status = OpCompilationStatus.CONVERTED_TO_TTNN
                    break
            except mp.queues.Empty:
                pass
            except Exception as e:
                process.terminate()
                raise e
            if time.time() - start > self.compiler_config.single_op_timeout:
                process.terminate()
                timeout_exceeded = True
                break
            if not process.is_alive():
                break
            time.sleep(0.01)
        process.join()

        if timeout_exceeded:
            msg = f"Timeout exceeded for op during compile after {self.compiler_config.single_op_timeout} seconds."
            print(msg, flush=True)
            binary = None

        return binary, op, msg

    def run_op(self, binary, *inputs):
        inputs = self.pre_process_inputs(*inputs)
        if not self.stderror_redirected:
            self.file_stderr = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
            self.stderror_redirected = True

        inputs_size = get_inputs_size(inputs)

        large_input = inputs_size >= self.op_memory_limit

        obj = {
            "binary": binary,
            "dump_file": self.file_stderr.name,
            "large_input": large_input,
        }

        inputs_file_path = None
        if large_input:
            obj["inputs"] = None
            try:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
                inputs_file_path = temp_file.name
                temp_file.close()

                with open(inputs_file_path, "wb") as f:
                    pickle.dump(inputs, f)

                obj["inputs_file_path"] = inputs_file_path
            except Exception as e:
                print(f"Error saving inputs to disk: {e}")
                if inputs_file_path and os.path.exists(inputs_file_path):
                    try:
                        os.remove(inputs_file_path)
                    except OSError:
                        pass
                large_input = False
        else:
            obj["inputs"] = inputs
            obj["inputs_file_path"] = None

        exec_event = mp.Event()
        if self.execute_process is None:
            self.execute_sender = mp.Queue()
            self.execute_receiver = mp.Queue()
            self.execute_process = mp.Process(
                target=execute_process,
                args=(self.execute_sender, self.execute_receiver, exec_event),
            )
            self.execute_process.start()
        self.execute_sender.put(obj)
        result = {}
        start = time.time()
        outputs = [None]
        timeout_exceeded = False
        while True:
            if not self.execute_process.is_alive():
                self.execute_process = None
                break
            try:
                result = self.execute_receiver.get_nowait()
                outputs = result["outputs"]
                exec_event.set()
                break
            except mp.queues.Empty:
                pass
            if time.time() - start > self.compiler_config.single_op_timeout:
                self.execute_process.terminate()
                self.execute_process = None
                timeout_exceeded = True
                break

        if inputs_file_path and os.path.isfile(inputs_file_path):
            try:
                os.remove(inputs_file_path)
            except OSError:
                pass

        if len(outputs) == 1:
            outputs = outputs[0]

        stderr_data = ""
        if outputs is None:
            file_stderr = open(self.file_stderr.name, "r")
            stderr_data = file_stderr.read()
            stderr_data = stderr_data.replace("\n", "\\n")
            stderr_data = re.sub(r"[^\x20-\x7E]", "", stderr_data)
            file_stderr.close()

            # If timeout is exceeded and stderr empty, add message and print to stdout.
            if timeout_exceeded and not stderr_data:
                stderr_data = f"Timeout exceeded for op during run after {self.compiler_config.single_op_timeout} seconds."
                print(stderr_data, flush=True)

        return outputs, stderr_data
