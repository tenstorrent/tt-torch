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
    OpByOpBackend,
    CompilerConfig,
    Op,
    OpCompilationStatus,
    RuntimeIntermediate,
)
from tt_torch.tools.utils import (
    run_model_proto,
    onnx_output_to_torch,
    torch_input_to_onnx,
)
from typing import Union


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
    ttir = tt_mlir.compile_stable_hlo_to_ttir(asm)
    sender.put({"ttir": ttir})
    ttir_event.wait()
    binary, ttnn = tt_mlir.compile_ttir_to_bytestream(ttir)
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
            outputs = tt_mlir.run_end_to_end(inputs, binary)

        sys.stderr = old_stderr
        sys.stdout = old_stdout
        file_stderr.close()

        sender.put({"outputs": outputs})
        exec_event.wait()

    sys.exit(0)


class Executor:
    def __init__(
        self,
        program: Union[torch.export.ExportedProgram, None] = None,
        graph_constants=None,
        compiler_config=None,
        required_pcc=0.99,
        required_atol=1e-2,
        device=None,
        async_mode=False,
    ):
        self.program = program
        self.binary = None
        if graph_constants is not None:
            self.graph_constants = (
                (graph_constants,)
                if isinstance(graph_constants, (int, float))
                else tuple(graph_constants)
            )
        else:
            self.graph_constants = None
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

        self.binary = None
        self.preprocessed_graph_constants = None
        self.device = device
        self.async_mode = async_mode
        self._validate_executor()

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

    def set_binary(self, binary):
        self.binary = binary

    def _get_device(self):
        if self.device is not None:
            return self.device
        # Return a default parent mesh
        device = tt_mlir.open_mesh_device([1, 1], tt_mlir.MeshDeviceOptions())
        return device

    def _cache_constants_if_needed(self, preprocessed_constants):
        if (
            self.compiler_config.cache_preprocessed_constants
            and self.graph_constants is not None
            and self.preprocessed_graph_constants is None
        ):
            self.preprocessed_graph_constants = preprocessed_constants

    def _cleanup_resources(self, preprocessed_activations, device):
        for t in preprocessed_activations:
            tt_mlir.deallocate_tensor(t, force=True)

        if self.device is None:
            tt_mlir.close_mesh_device(device)

    def _generate_golden_intermediate_cache(self, gm, inputs):
        print("Generating golden intermediate cache")
        node_to_tensor = {}
        input_index = 0
        outputs = []
        num_nodes = len(gm.graph.nodes)
        out_degree = {}
        for idx, node in enumerate(gm.graph.nodes):
            print(f"Compiling {idx}/{num_nodes}: <{node.op}>{node.name}\t{node.target}")
            out_degree[node] = len(node.users)
            if node.op == "placeholder":
                node_to_tensor[node] = inputs[input_index]
                input_index += 1
            elif node.op == "get_attr":
                for buffer in gm.named_buffers():
                    if buffer[0] == node.target:
                        node_to_tensor[node] = buffer[1]
                        break
            elif node.op == "call_function":
                args = []
                for arg in node.args:
                    if isinstance(arg, torch.fx.node.Node):
                        args.append(node_to_tensor[arg])
                    elif isinstance(arg, list):
                        args.append(
                            [
                                node_to_tensor[a]
                                if isinstance(a, torch.fx.node.Node)
                                else a
                                for a in arg
                            ]
                        )
                    else:
                        args.append(arg)

                golden = node.target(*args, **node.kwargs)

                # some ops return scalar (0D tensor) as output (e.g. aten.select.int)
                if isinstance(golden, torch.Tensor) and golden.dim() == 0:
                    print(f"Unsqueezing golden {golden} to {golden.unsqueeze(0)}")
                    golden = golden.unsqueeze(0)

                # some ops return a tuple of tensors as output (e.g. max_pool_2d_with_indices)
                # we expect to only use the first, though this may be changed in the future
                elif isinstance(golden, (tuple, list)) and len(golden) > 1:
                    golden = golden[0]
                    print(
                        f"\033[33m[WARNING] {node.name} has {len(golden)} outputs, but we can only get one from runtime.\033[0m"
                    )
                cache_entry = RuntimeIntermediate(node, golden)
                self.compiler_config.runtime_intermediate_cache[node.name] = cache_entry
                print(f"Caching runtime intermediate for {node.name}")
                tensor = node.target(*args, **node.kwargs)
                node_to_tensor[node] = tensor

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
                self.program.graph_module != None
            ), "Cannot run base executor without torch graph"
            return self.program.graph_module(
                *(self.graph_constants + tuple(self.program.buffers()) + inputs)
            )

        assert self.binary is not None
        if self.compiler_config.typecast_inputs:
            inputs = self.typecast_inputs(inputs)

        activations_len = len(inputs)
        if (
            self.graph_constants is not None
            and self.preprocessed_graph_constants is None
        ):
            inputs = self.graph_constants + inputs

        inputs = list(inputs)
        device = self._get_device()

        binary = tt_mlir.create_binary_from_bytestream(self.binary)
        program_idx = 0

        tensor_start_idx = 0
        if self.preprocessed_graph_constants is not None:
            tensor_start_idx = len(self.preprocessed_graph_constants)

        preprocessed_inputs = tt_mlir.preprocess_inputs(
            device, inputs, binary, program_idx, tensor_start_idx
        )

        if self.preprocessed_graph_constants is not None:
            preprocessed_inputs = (
                self.preprocessed_graph_constants + preprocessed_inputs
            )

        if self.compiler_config._enable_intermediate_verification:
            # put this as close to the binding as possible to ensure the GM is not mutated past this point
            self._generate_golden_intermediate_cache(self.program, inputs)

        if self.async_mode:
            outputs = tt_mlir.run_async(
                device, binary, program_idx, preprocessed_inputs
            )
        else:
            outputs = tt_mlir.run(device, binary, program_idx, preprocessed_inputs)

        self._cache_constants_if_needed(preprocessed_inputs[:-activations_len])
        self._cleanup_resources(preprocessed_inputs[-activations_len:], device)

        return outputs


class OnnxExecutor(Executor):
    def __init__(self, model_proto: onnx.ModelProto):
        self.model_proto = model_proto
        self.binary = None
        self.sess = None
        self.device = None

    def typecast_inputs(self, inputs):
        raise NotImplementedError("This should not be called on an OnnxExecutor.")

    def __call__(self, *inputs):
        if self.binary is None:
            # Only want to load the model proto into one inference session
            # since models can be big
            output = run_model_proto(
                sess=self.sess, model_proto=self.model_proto, inputs=inputs
            )
            return onnx_output_to_torch(output)

        return tt_mlir.run_end_to_end(inputs, self.binary)


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
        compiler_config=None,
        required_pcc=0.99,
        required_atol=1e-2,
        device=None,
        async_mode=False,
    ):
        super().__init__(
            program=None,
            graph_constants=None,
            compiler_config=compiler_config,
            required_pcc=required_pcc,
            required_atol=required_atol,
            device=device,
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
        obj = {"asm": module.operation.get_asm()}
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
