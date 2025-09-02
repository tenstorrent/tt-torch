# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import time
import operator

from .xla_decompositions import (
    CUSTOM_DECOMPOSITION_TABLE,
)
import os
import tempfile
import multiprocessing as mp
import re
import pickle
import faulthandler
import collections
from ..passes import (
    bypass_redundant_getitem,
    bypass_dtype_promotion,
    bypass_redundant_cast,
    rectify_buffer_inplace_copy,
    run_shape_prop,
    constant_fold,
)

from torch.export.graph_signature import InputKind
from torch._dynamo import register_backend

from ..backend import BackendOptions
from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    Op,
    OpCompilationStatus,
    calculate_atol,
    calculate_pcc,
)

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import tt_torch.dynamo.sharding_utils as ts

from ..executor import get_inputs_size, gb_to_bytes


class XLATensorCache:
    """Singleton cache for tensors that have been moved to XLA device."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
            cls._instance._debug = os.getenv("DEBUG_XLA_CACHE", "0") == "1"
        return cls._instance
    
    def _get_cache_key(self, tensor):
        """Generate a composite cache key based on tensor properties."""
        try:
            # Use tensor's memory address (id), shape, dtype, and device as key
            tensor_id = id(tensor)
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            device = str(tensor.device)
            _tensor = tensor # for rev lookup
                 
            cache_key = (tensor_id, shape, dtype, device, _tensor)
            
            # if self._debug:
            #     print(f"[XLA Cache] Generated cache key: {cache_key}", flush=True)
                
            return cache_key
        except Exception as e:
            if self._debug:
                print(f"[XLA Cache] Error generating cache key: {e}", flush=True)
            # Fallback to tensor object itself if key generation fails
            return tensor
    
    def get(self, tensor):
        """Get tensor from cache if it exists."""
        cache_key = self._get_cache_key(tensor)
        cached_tensor = self._cache.get(cache_key)
        
        if cached_tensor is not None and self._debug:
            try:
                xla_tensor_id = torch_xla._XLAC._xla_get_tensor_id(cached_tensor)
                xla_sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(cached_tensor)
                print(f"[XLA Cache] Retrieved tensor from cache: key={cache_key}, xla_tensor_id={xla_tensor_id}, xla_sharding_spec={xla_sharding_spec}, cached tensor device {cached_tensor.device}", flush=True)
            except:
                print(f"[XLA Cache] Retrieved tensor from cache: key={cache_key}", flush=True)
                
        return cached_tensor
    
    def put(self, original_tensor, cached_tensor):
        """Add tensor to cache."""
        cache_key = self._get_cache_key(original_tensor)
        
        if self._debug:
            try:
                # sometimes non xla tensors have this called on them -> cached tensor is somehow not an XLA tensor. 
                xla_tensor_id = torch_xla._XLAC._xla_get_tensor_id(cached_tensor)
                xla_sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(cached_tensor)
                print(f"[XLA Cache] Adding tensor to cache: key={cache_key}, xla_tensor_id={xla_tensor_id}, xla_sharding_spec={xla_sharding_spec}", flush=True)
            except:
                print(f"[XLA Cache] Adding tensor to cache: key={cache_key}", flush=True)
                
        self._cache[cache_key] = cached_tensor
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
    
    def size(self):
        """Get the number of items in cache."""
        return len(self._cache)
    
    def get_key_for_device_tensor(self, cached_tensor):
        """Get the cache key for a given cached tensor value (reverse lookup).
        
        Args:
            cached_tensor: The tensor that was previously cached
            
        Returns:
            The cache key if found, None otherwise
        """
        for key, value in self._cache.items():
            if value is cached_tensor:
                if self._debug:
                    print(f"[XLA Cache] Found key for cached tensor: {key}", flush=True)
                return key
        
        if self._debug:
            print(f"[XLA Cache] No key found for cached tensor", flush=True)
        return None


def cast_ios_and_run(node, args, kwargs):
    try:
        out_df = node.meta["tensor_meta"].dtype
        out_df_known = True
    except Exception:
        out_df_known = False

    if out_df_known:
        cast_args = [
            arg.to(torch.float32)
            if isinstance(arg, torch.Tensor) and torch.is_floating_point(arg)
            else arg
            for arg in args
        ]
        golden = node.target(*cast_args, **kwargs)
        golden = golden.to(out_df)
    else:
        golden = node.target(*args, **kwargs)
    return golden


def execute_pjrt_process(receiver, sender, exec_event):
    while 1:
        obj = receiver.get()
        faulthandler.disable()
        gm = obj["gm"]
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

        from typing import Union, List, Dict

        def push_tensors_to_device(
            tensors, device, detach=False
        ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
            if isinstance(tensors, torch.Tensor):
                out = tensors.to(device)
                if detach:
                    out = out.detach()
                return out
            elif isinstance(tensors, (list, tuple)):
                return [
                    push_tensors_to_device(tensor, device, detach) for tensor in tensors
                ]
            elif isinstance(tensors, dict):
                return {
                    k: push_tensors_to_device(v, device, detach)
                    for k, v in tensors.items()
                }
            else:
                return tensors

        def flatten_inputs(inputs):
            if isinstance(inputs, (list, tuple)):
                flattened = []
                for inp in inputs:
                    flattened.extend(flatten_inputs(inp))
                return flattened
            elif isinstance(inputs, dict):
                return {k: flatten_inputs(v) for k, v in inputs.items()}
            else:
                return [inputs]

        outputs = None
        if inputs is not None:
            inputs = push_tensors_to_device(inputs, device=xm.xla_device())
            inputs = flatten_inputs(inputs)
            gm = gm.to(xm.xla_device())
            outputs = gm(*inputs)
            xm.mark_step()
            outputs = push_tensors_to_device(outputs, device="cpu", detach=True)

        sender.put({"outputs": outputs})
        exec_event.wait()


class XLAOpByOpExecutor:

    # Class attributes for identifying each op w/ unique incrementing id
    # across graph breaks, and for running just a specific op.
    global_op_idx = 0
    run_global_op_idx = None
    compiling_time = 0.0
    running_time = 0.0
    golden_time = 0.0

    def __init__(self, gm, compiler_config, required_pcc=0.99, required_atol=1e-2):
        self.gm = gm
        self.compiler_config = compiler_config

        self.required_pcc = required_pcc
        self.required_atol = required_atol

        # Debug mode to run only specific op given global_op_idx
        if XLAOpByOpExecutor.run_global_op_idx is None:
            run_global_op_idx_env = os.getenv("RUN_GLOBAL_OP_IDX")
            XLAOpByOpExecutor.run_global_op_idx = (
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
        self.op_memory_limit = gb_to_bytes(0.5)  # 512MB limi

    def is_node_valid(self, node):
        if not isinstance(node.target, torch._ops.OpOverload):
            if "getitem" not in node.name:
                raise ValueError(f"Node target is not an OpOverload: {node.name}")
            return False
        return True

    def get_node_name(self, node):
        name = node.target.name() if hasattr(node.target, "name") else node.name
        return name

    def get_input_shapes_and_constants(self, *inputs):
        input_shapes_and_constants = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                input_shapes_and_constants.append(inp.shape)
            elif isinstance(inp, (list, tuple, torch.nn.ParameterList)):
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

    def get_single_op_graph_module(self, node, inputs, **kwargs):
        input_shapes_and_constants = self.get_input_shapes_and_constants(*inputs)

        name = node.target.name() if hasattr(node.target, "name") else node.name
        if not isinstance(node.target, torch._ops.OpOverload):
            if "getitem" not in name:
                raise ValueError(f"Node target is not an OpOverload: {name}")
            return None, None

        if name == "aten::copy_":
            raise ValueError(f"inline ops are not supported: {name}")
            return None, None

        # Skip validation ops (like aten._assert_tensor_metadata) that lack tensor metadata
        if "tensor_meta" not in node.meta:
            print(f"Warning: {node.target} missing tensor_meta, skipping compile.")
            return None, None

        op = Op(name, input_shapes_and_constants, self.compiler_config.model_name)
        if op.unique_key() not in self.compiler_config.unique_ops:
            op.global_op_idx = XLAOpByOpExecutor.global_op_idx
            op.model_group = self.compiler_config.model_group
            self.compiler_config.unique_ops[op.unique_key()] = op
        else:
            self.compiler_config.unique_ops[op.unique_key()].num_ops += 1
            return None, None

        graph = torch.fx.Graph()

        def generate_placeholders(args, prefix="input_"):
            placeholders = []
            for input_idx, inp in enumerate(args):
                if isinstance(inp, (list, tuple)):
                    sub_placeholders = generate_placeholders(
                        inp, prefix=f"{prefix}{input_idx}_"
                    )
                    placeholders.append(sub_placeholders)
                else:
                    placeholders.append(graph.placeholder(f"{prefix}{input_idx}"))
            return placeholders

        placeholders = generate_placeholders(inputs)

        if len(placeholders) != len(node.args):
            # are any of the args duplicates? If so, we need to duplicate the placeholders
            for idx, arg in enumerate(node.args):
                if arg in node.args[idx + 1 :]:
                    placeholders.append(placeholders[idx])

        placeholders = tuple(placeholders)

        graph_node = graph.call_function(node.target, placeholders, kwargs)
        graph_node.meta["tensor_meta"] = node.meta["tensor_meta"]

        # if the node has multiple outputs, add a getitem for each and append to graph
        if not isinstance(
            node.meta["tensor_meta"], torch.fx.passes.shape_prop.TensorMetadata
        ):
            getitem_nodes = []
            graph_node.meta["val"] = node.meta["val"]

            # if the output of the getitem node is not used, we don't append it to the graph
            for user in node.users:
                assert user.target == operator.getitem
                if len(user.users) == 0:
                    continue

                idx = user.args[1]
                getitem_node = graph.call_function(
                    operator.getitem, args=(graph_node, idx)
                )
                getitem_nodes.append(getitem_node)
                tensor_meta = node.meta["tensor_meta"][idx]
                getitem_node.meta["tensor_meta"] = tensor_meta
            out = graph.output(tuple(getitem_nodes))
        else:
            out = graph.output((graph_node,))
        if "tensor_meta" not in node.meta:
            raise ValueError(f"Node {node} does not have tensor_meta")

        op.compilation_status = OpCompilationStatus.CREATED_GRAPH
        out.meta["tensor_meta"] = node.meta["tensor_meta"]

        out_meta = out.meta["tensor_meta"]
        if isinstance(out_meta, torch.fx.passes.shape_prop.TensorMetadata):
            out_meta = (out_meta,)
        for out in out_meta:
            op.output_shapes.append([dim for dim in out.shape])

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        return gm, op

    # Helper function to print markers
    def print_marker(self, msg, idx, num_nodes, op_info, error="", time=0.0):
        print(
            f"{msg:<10} global_op_idx: {XLAOpByOpExecutor.global_op_idx} ({idx}/{num_nodes}): {op_info} | time: {time:.4f} s | {error}",
            flush=True,
        )

    def should_test_op(self):
        return (
            XLAOpByOpExecutor.run_global_op_idx is None
            or XLAOpByOpExecutor.global_op_idx == XLAOpByOpExecutor.run_global_op_idx
        )

    # Helper function to get extract exception source for printing concise message
    def get_exception_source(self, e):
        import traceback

        filename, lineno, function, _ = traceback.extract_tb(e.__traceback__)[-1]
        return f"{type(e).__name__}: {e} in {filename.split('/')[-1]}:{lineno} ({function})"

    def run_op(self, gm, *inputs):
        if not self.stderror_redirected:
            self.file_stderr = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
            self.stderror_redirected = True

        file_stderr = open(self.file_stderr.name, "w")
        old_stdout = os.dup(1)  # Backup stdout descriptor.
        old_stderr = os.dup(2)  # Backup stderr descriptor.
        os.dup2(file_stderr.fileno(), 1)  # Redirect stdout (fd 1)
        os.dup2(file_stderr.fileno(), 2)  # Redirect stderr (fd 2)

        inputs_size = get_inputs_size(inputs)

        large_input = inputs_size >= self.op_memory_limit

        obj = {
            "gm": gm,
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
                target=execute_pjrt_process,
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

        # Revert redirection of stdout/stderr.
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        file_stderr.close()

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

    def run_gm_op_by_op(self, *user_inputs):
        node_to_tensor = {}
        inputs = user_inputs
        input_index = 0
        outputs = []
        num_nodes = len(self.gm.graph.nodes)
        out_degree = {}

        for idx, node in enumerate(self.gm.graph.nodes):
            self.print_marker("\nProcessing", idx, num_nodes, node.target)

            out_degree[node] = len(node.users)
            if node.op == "placeholder":
                node_to_tensor[node] = inputs[input_index]
                input_index += 1
            elif node.op == "get_attr":
                if node.target in self.gm.state_dict():
                    node_to_tensor[node] = self.gm.state_dict()[node.target]
                elif hasattr(self.gm, node.target):
                    node_to_tensor[node] = getattr(self.gm, node.target)
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

                op = None

                test_this_op = self.should_test_op()
                # Another useful debug method:
                # test_this_op = str(node.target) == "aten.convolution.default"

                if test_this_op:
                    try:
                        start = time.time()
                        gm, op = self.get_single_op_graph_module(node, args)
                        end = time.time()
                        self.print_marker(
                            "Compiling", idx, num_nodes, node.target, time=(end - start)
                        )
                        XLAOpByOpExecutor.compiling_time += end - start

                        self.set_runtime_stack_dump(None, op)

                    except Exception as e:
                        gm = None
                        e_msg = self.get_exception_source(e)
                        self.print_marker(
                            "Failed to compile", idx, num_nodes, node.target, e_msg
                        )
                start = time.time()
                golden = cast_ios_and_run(node, args, node.kwargs)
                end = time.time()
                self.print_marker(
                    "Golden", idx, num_nodes, node.target, time=(end - start)
                )
                XLAOpByOpExecutor.golden_time += end - start
                calculated = None
                if (
                    self.compiler_config.compile_depth == CompileDepth.EXECUTE_OP_BY_OP
                    and test_this_op
                    and gm is not None
                ):

                    try:
                        start = time.time()
                        calculated, stderr = self.run_op(gm, *args)

                        end = time.time()
                        self.print_marker(
                            "Running", idx, num_nodes, node.target, time=(end - start)
                        )
                        XLAOpByOpExecutor.running_time += end - start
                        self.set_runtime_stack_dump(stderr, op)

                        if calculated is None:
                            op.compilation_status = (
                                OpCompilationStatus.CONVERTED_TO_TTNN
                            )
                            raise ValueError(f"Failed to execute: \n {stderr}")
                        op.compilation_status = OpCompilationStatus.EXECUTED
                        if self.compiler_config.verify_op_by_op:
                            atol = calculate_atol(calculated, golden)
                            op.atol = atol
                            if atol > self.required_atol:
                                print(f"atol too high for {idx}: {atol}")
                            pcc = calculate_pcc(calculated, golden)
                            op.pcc = pcc
                            if pcc < self.required_pcc:
                                print(f"pcc too low for {idx}: {pcc}")
                    except Exception as e:
                        e_msg = self.get_exception_source(e)
                        self.print_marker(
                            "Failed to execute", idx, num_nodes, node.target, stderr
                        )

                if out_degree[node] > 0:
                    node_to_tensor[node] = golden
            elif node.op == "output":
                args = node.args[0]
                output_tensors = [node_to_tensor[arg] for arg in args]
                outputs = output_tensors

            args_set = set()
            for arg in node.args:
                if arg in args_set:
                    continue
                args_set.add(arg)
                if isinstance(arg, torch.fx.node.Node):
                    out_degree[arg] -= 1
                    if out_degree[arg] == 0 and arg.op != "output":
                        del node_to_tensor[arg]
                        out_degree.pop(arg)

            # Finished handling this op, increment global op index
            XLAOpByOpExecutor.global_op_idx += 1

        self.compiler_config.save_unique_ops()
        if self.execute_process is not None:
            self.execute_process.terminate()
            self.execute_process = None
        if self.stderror_redirected:
            os.unlink(self.file_stderr.name)
            self.stderror_redirected = False
        print(
            f"Total Time - Compiling: {XLAOpByOpExecutor.compiling_time:.2f} s, Running: {XLAOpByOpExecutor.running_time:.2f} s, Golden: {XLAOpByOpExecutor.golden_time:.2f} s"
        )
        return outputs

    def __call__(self, *inputs):
        return self.run_gm_op_by_op(*inputs)


def bypass_assert_tensor_metadata(gm):
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten._assert_tensor_metadata.default
        ):
            gm.graph.erase_node(node)
    return gm


def xla_pass_pipeline(gm, example_inputs, compiler_config):
    decompositions = torch._decomp.core_aten_decompositions()
    decompositions.update(CUSTOM_DECOMPOSITION_TABLE)
    compiled_graph = (
        torch.export.export_for_training(gm, tuple(example_inputs), strict=False)
        .run_decompositions(decompositions)
        .module()
    )

    compiled_graph = bypass_dtype_promotion(compiled_graph, compiler_config)
    run_shape_prop(compiled_graph, example_inputs)
    compiled_graph = bypass_redundant_cast(compiled_graph)

    if compiler_config.enable_consteval:
        compiled_graph = constant_fold(compiled_graph)
    elif compiler_config.consteval_parameters:
        raise Exception("consteval_parameters is enabled but enable_consteval is not")
    
    # compiled_graph = rectify_buffer_inplace_copy(compiled_graph)
    compiled_graph = bypass_redundant_getitem(compiled_graph)
    compiled_graph = bypass_assert_tensor_metadata(compiled_graph)
    program = torch.export.export(compiled_graph, tuple(example_inputs), strict=False)

    return program


class XLAExecutor:
    def __init__(self, program, compiler_config):
        self._tensor_cache = XLATensorCache()
        self.program = program
        self.compiler_config = compiler_config
        self.arg_type_map_str = None

        self.inputs = []
        self.user_input_indices = []
        
        print(f"[XLA Debug] Processing {len(self.program._graph_signature.input_specs)} input specs:", flush=True)
        for idx, input_spec in enumerate(self.program._graph_signature.input_specs):
            print(f"[XLA Debug] Input {idx}: kind={input_spec.kind}, arg={input_spec.arg}, target={getattr(input_spec, 'target', 'N/A')}", flush=True)
            
            if input_spec.kind == InputKind.USER_INPUT:
                # print(f"[XLA Debug] Input {idx} is USER_INPUT, adding None placeholder", flush=True)
                self.inputs.append(None)
                self.user_input_indices.append(idx)
            else:
                source_tensor = self.program.state_dict[input_spec.target]
                # print(f"[XLA Debug] Input {idx} is {input_spec.kind}, source tensor shape: {source_tensor.shape}, dtype: {source_tensor.dtype}, device: {source_tensor.device}", flush=True)

                cached_tensor = self._tensor_cache.get(source_tensor)                
                if cached_tensor is not None:
                    # Why does shard spec get lost on the kv cache tensors? I believe it to be an inplace annotation 
                    # shard_spec = ts.get_sharding(source_tensor)
                    # if shard_spec is not None:
                    #     print(f"[XLA Debug] [James] Re-apply sharding to cache retreived tensor | applying shard spec: {shard_spec} with mesh: {self.compiler_config.mesh}", flush=True)
                    #     xs.mark_sharding(
                    #         cached_tensor, self.compiler_config.mesh, shard_spec
                    #     )
                    self.inputs.append(cached_tensor)
                else:
                    shard_spec = ts.get_sharding(source_tensor)
                    print(
                        # f"[XLA Debug] Input {idx} not in cache - shape: {source_tensor.shape}, shard spec: {shard_spec}, moving to XLA device", flush=True
                    )
                    # immediate allocation of host tensor
                    device_tensor = source_tensor.to("xla")  
                    if shard_spec is not None:
                        # print(f"[XLA Debug] Input {idx} applying shard spec: {shard_spec} with mesh: {self.compiler_config.mesh}", flush=True)
                        xs.mark_sharding(
                            device_tensor, self.compiler_config.mesh, shard_spec
                        )
                    else:
                        # print(f"[XLA Debug] Input {idx} no sharding applied", flush=True)
                        pass
                    self.inputs.append(device_tensor)
                    
                    self._tensor_cache.put(source_tensor, device_tensor)
                    # print(f"[XLA Debug] Input {idx} cached tensor with device: {device_tensor.device}", flush=True)
        
        print(f"[XLA Debug] Initialization complete - total inputs: {len(self.inputs)}, user input indices: {self.user_input_indices}", flush=True)

    def push_tensors_to_device(self, inputs, device):
        if hasattr(inputs, "to"):
            if device not in [inputs.device, inputs.device.type]:
                cached_tensor = self._tensor_cache.get(inputs)
                if cached_tensor is not None:
                    return cached_tensor
                else:
                    shard_spec = ts.get_sharding(inputs)
                    inp = inputs.to(device)
                    if shard_spec is not None:
                        xs.mark_sharding(inp, self.compiler_config.mesh, shard_spec)
                    self._tensor_cache.put(inputs, inp)
                    return inp
            else:
                return inputs
        elif isinstance(
            inputs, dict
        ):  # transformers input/output objects are subclasses of dict, however we still wish to return the same wrapper object
            return type(inputs)(
                **{k: self.push_tensors_to_device(v, device) for k, v in inputs.items()}
            )
        elif isinstance(inputs, collections.abc.Sequence):
            return type(inputs)(
                [self.push_tensors_to_device(i, device) for i in inputs]
            )
        elif hasattr(inputs, "key_cache") or hasattr(inputs, "value_cache"):
            if hasattr(inputs, "key_cache"):
                inputs.key_cache = self.push_tensors_to_device(inputs.key_cache, device)
            if hasattr(inputs, "value_cache"):
                inputs.value_cache = self.push_tensors_to_device(
                    inputs.value_cache, device
                )
            return inputs
        else:
            return inputs

    def rectify_shard_specs_during_call(self, inputs:list[torch.Tensor]):
        # some shard specs get changed for the buffer tensors, for some unknown reason after mark_step()
        # thus we remark the shard specs before the next call.
        for xla_tensor in inputs:
            maybe_key = self._tensor_cache.get_key_for_device_tensor(xla_tensor)
            if maybe_key:
                _,_,_,_,cpu_tensor = maybe_key
                shard_spec = ts.get_sharding(cpu_tensor)
                if shard_spec:
                    xs.mark_sharding(xla_tensor, self.compiler_config.mesh, shard_spec)
    
    def generate_arg_type_map_str(self, output_object):
        hlo_input_ids, _ = torch_xla._XLAC._get_tensors_xla_device_data_node(
            output_object
        )

        # xm.get_stablehlo(output_object) gives a graph with just as many inputs as in hlo_input_ids

        hlo_input_positions = [id - min(hlo_input_ids) for id in hlo_input_ids]

        def get_kind_str(kind):
            if kind == InputKind.USER_INPUT:
                return "input"
            elif kind == InputKind.PARAMETER:
                return "parameter"
            else:
                return "constant"

        arg_types = []
        output_args = [o.arg for o in self.program.graph_signature.output_specs]
        for idx in range(len(hlo_input_positions)):
            if hlo_input_positions[idx] < len(self.program.graph_signature.input_specs):
                in_spec = self.program.graph_signature.input_specs[
                    hlo_input_positions[idx]
                ]

                # If an input is passed right through to the output, it will not be
                # captured as an argument
                if in_spec.arg in output_args:
                    continue

                arg_types.append(get_kind_str(in_spec.kind))
            else:
                arg_types.append("constant")

        self.arg_type_map_str = "main=" + ",".join(arg_types)

    def __call__(self, *args):
        args = self.push_tensors_to_device(args, "xla")
        inputs = self.inputs

        # self.program.graph_module.graph.print_tabular()
        
        self.rectify_shard_specs_during_call(inputs)

        # Print input names and shapes from graph signature
        # for idx, input_spec in enumerate(self.program.graph_signature.input_specs):
        #     if input_spec.kind == InputKind.USER_INPUT:
        #         shape = (
        #             args[self.user_input_indices.index(idx)].shape
        #             if idx in self.user_input_indices
        #             else "N/A"
        #         )
        #     else:
        #         shape = (
        #             self.program.state_dict[input_spec.target].shape
        #             if input_spec.target in self.program.state_dict
        #             else "N/A"
        #         )
        #     xla_tensor_id = torch_xla._XLAC._xla_get_tensor_id(inputs[idx]) if inputs[idx]!=None else "N/A"
        #     print(f"During __call__: Input {input_spec.arg}: kind={input_spec.kind}, shape={shape}, torchxla tensor id: {xla_tensor_id}", flush=True)

        for idx in range(len(args)):
            inputs[self.user_input_indices[idx]] = args[idx]

        output = self.program.graph_module(*inputs)

        if self.compiler_config.arg_type_map_override:
            if self.arg_type_map_str is None:
                self.generate_arg_type_map_str(output)
            if os.environ.get("ARG_TYPE_MAP_OVERRIDE") != self.arg_type_map_str:
                os.environ["ARG_TYPE_MAP_OVERRIDE"] = self.arg_type_map_str
        
        torch_xla.sync()
        if self.compiler_config.push_outputs_to_cpu:
            output = self.push_tensors_to_device(output, "cpu")
        
        dump_static_cache = True
        if dump_static_cache:
            for i,_input in enumerate(inputs):
                is_static_cache = _input.dim()==4 and _input.shape[2] == 128 and _input.shape[3]==128
                print(f"input {i}: device = {_input.device}, shape {_input.shape} and shard spec {torch_xla._XLAC._get_xla_sharding_spec(_input)}", flush=True)
                if is_static_cache:         
                    _input = _input.to('cpu')
                    try:
                        mean_val = torch.mean(_input[0,0,:,:], dim=-1)
                        print(f"[STATIC CACHE DUMP ]mean along seqlen for static cache @ input idx {i} and shape {_input.shape}:", mean_val, flush=True)
                    except Exception as e:
                        print(f"\tWarning: Could not compute mean for static cache {i}: {e}")
                        print(f"\tmean along seqlen for static cache @ input idx {i} and shape {_input.shape}: <error during computation>")
        
        return output

    def __del__(self):
        # Remove the arg type map override environment variable
        os.environ.pop("ARG_TYPE_MAP_OVERRIDE", None)


@register_backend(name="tt-experimental")
def xla_backend(gm, example_inputs, options: BackendOptions = None):
    print("Note: Using experimental XLA backend.")

    if options is None:
        cc = CompilerConfig()
    else:
        cc = options.compiler_config

    program = xla_pass_pipeline(gm, example_inputs, cc)

    if cc.compile_depth == CompileDepth.EXECUTE_OP_BY_OP:
        return XLAOpByOpExecutor(program.module(), cc)
    return XLAExecutor(program, cc)
