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
import sys
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

from ..executor import get_inputs_size, gb_to_bytes


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

        sys.stderr = old_stderr
        sys.stdout = old_stdout
        file_stderr.close()

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

        inputs_size = get_inputs_size(inputs)

        large_input = inputs_size >= self.op_memory_limit

        obj = {
            "gm": gm,
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
                            "Failed to execute", idx, num_nodes, node.target, e_msg
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

    compiled_graph = bypass_redundant_getitem(compiled_graph)
    print("[James] disable rectify buffer inplace copy")
    # compiled_graph = rectify_buffer_inplace_copy(compiled_graph) 
    compiled_graph = bypass_assert_tensor_metadata(compiled_graph)
    program = torch.export.export(compiled_graph, tuple(example_inputs), strict=False)

    return program


class XLAExecutor:
    # Shared cache for previously seen inputs across all XLA executor instances
    previously_seen_inputs = set()
    # Cache for device-transferred tensors based on object identity and target
    device_tensor_cache = {}
    
    def __init__(self, program, compiler_config):
        self.program = program
        self.compiler_config = compiler_config
        self.arg_type_map_str = None
        self.arg_ref_map_str = None

        self.inputs = []
        self.user_input_indices = []
        for idx, input_spec in enumerate(self.program._graph_signature.input_specs):
            if input_spec.kind == InputKind.USER_INPUT:
                self.inputs.append(None)
                self.user_input_indices.append(idx)
            else:
                # Create cache key based on object identity and target
                source_tensor = self.program.state_dict[input_spec.target]
                cache_key = (id(source_tensor), input_spec.target)
                
                # Check if we already have this tensor on device
                if cache_key in XLAExecutor.device_tensor_cache:
                    cached_tensor = XLAExecutor.device_tensor_cache[cache_key]
                    self.inputs.append(cached_tensor)
                    print(f"[Cache] Reusing cached tensor for {input_spec.target}")
                else:
                    # Move to device and cache it
                    device_tensor = source_tensor.to("xla")
                    XLAExecutor.device_tensor_cache[cache_key] = device_tensor
                    self.inputs.append(device_tensor)
                    print(f"[Cache] Cached new tensor for {input_spec.target}")
        # import pdb; pdb.set_trace()
        
        
    def _create_tensor_cache_key(self, tensor):
        """Create a unique cache key for a tensor based on its properties."""
        if not isinstance(tensor, torch.Tensor):
            return None
        
        # Use tensor id, shape, dtype, and device as cache key
        return (id(tensor), tuple(tensor.shape), tensor.dtype, str(tensor.device))
    
    def _get_tensor_name(self, tensor, input_idx):
        """Generate a descriptive name for the tensor."""
        if not isinstance(tensor, torch.Tensor):
            return f"input_{input_idx}_non_tensor"
        
        return f"input_{input_idx}_shape_{list(tensor.shape)}_dtype_{tensor.dtype}"
    
    def _check_and_log_previously_seen_inputs(self, inputs):
        """Check which inputs have been previously seen and log them with detailed cache behavior analysis."""
        cache_hits = []
        new_inputs = []
        static_cache_tensors = []
        
        print(f"\n[XLA Cache] === DETAILED CACHE ANALYSIS ===")
        print(f"[XLA Cache] Total inputs to analyze: {len(inputs)}")
        print(f"[XLA Cache] Current cache contains {len(XLAExecutor.previously_seen_inputs)} unique tensor keys")
        
        # Show all existing cache keys first
        if XLAExecutor.previously_seen_inputs:
            print(f"[XLA Cache] Existing cache keys:")
            for i, cache_key in enumerate(sorted(XLAExecutor.previously_seen_inputs)):
                tensor_id, shape, dtype, device = cache_key
                is_static_cache = len(shape) == 4 and shape == (1, 8, 128, 128)
                cache_type = "[STATIC CACHE]" if is_static_cache else "[REGULAR]"
                print(f"[XLA Cache]   {i+1:2d}. {cache_type} id={tensor_id}, shape={shape}, dtype={dtype}, device={device}")
        
        for idx, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                cache_key = self._create_tensor_cache_key(inp)
                tensor_name = self._get_tensor_name(inp, idx)
                
                # Detect static cache tensors (1x8x128x128)
                is_static_cache = inp.dim() == 4 and inp.shape == (1, 8, 128, 128)
                if is_static_cache:
                    static_cache_tensors.append((idx, inp, cache_key, tensor_name))
                
                print(f"[XLA Cache] Input {idx}: {tensor_name}")
                print(f"[XLA Cache]   Cache key: id={cache_key[0]}, shape={cache_key[1]}, dtype={cache_key[2]}, device={cache_key[3]}")
                print(f"[XLA Cache]   Tensor object id: {id(inp)}")
                print(f"[XLA Cache]   XLA tensor id: {torch_xla._XLAC._xla_get_tensor_id(inp)}")
                
                if is_static_cache:
                    print(f"[XLA Cache]   *** STATIC CACHE TENSOR DETECTED ***")
                    # # Analyze tensor content for debugging
                    # try:
                    #     mean_val = torch.mean(inp[0, 0, :, :], dim=-1)
                    #     nonzero_indices = torch.nonzero(mean_val, as_tuple=False).squeeze(-1)
                    #     nonzero_count = len(nonzero_indices)
                    #     print(f"[XLA Cache]   Content analysis: nonzero_count={nonzero_count}, first_few_nonzero={nonzero_indices[:5].tolist() if nonzero_count > 0 else []}")
                    # except Exception as e:
                    #     print(f"[XLA Cache]   Content analysis failed: {e}")
                
                if cache_key in XLAExecutor.previously_seen_inputs:
                    cache_hits.append((idx, tensor_name))
                    print(f"[XLA Cache]   Result: CACHE HIT ✓")
                else:
                    new_inputs.append((idx, tensor_name))
                    XLAExecutor.previously_seen_inputs.add(cache_key)
                    print(f"[XLA Cache]   Result: CACHE MISS - added to cache")
                    
                    # Detailed cache miss analysis for static cache tensors
                    if is_static_cache:
                        print(f"[XLA Cache]   *** CACHE MISS ANALYSIS FOR STATIC CACHE ***")
                        print(f"[XLA Cache]   Reason: Each tensor object has unique id() = {id(inp)}")
                        print(f"[XLA Cache]   Even identical static cache tensors get different cache keys due to object identity")
                        
                        # Check if we have any other static cache tensors in the cache
                        static_cache_count = sum(1 for key in XLAExecutor.previously_seen_inputs 
                                               if len(key[1]) == 4 and key[1] == (1, 8, 128, 128))
                        print(f"[XLA Cache]   Total static cache tensors in cache: {static_cache_count}")
                        
                        # Show similar cache keys (same shape/dtype/device but different id)
                        similar_keys = [key for key in XLAExecutor.previously_seen_inputs 
                                      if key[1] == cache_key[1] and key[2] == cache_key[2] and key[3] == cache_key[3]]
                        if similar_keys:
                            print(f"[XLA Cache]   Similar cache keys with same shape/dtype/device:")
                            for sim_key in similar_keys:
                                print(f"[XLA Cache]     id={sim_key[0]} (different from current {cache_key[0]})")
                print(f"[XLA Cache]   ---")
        
        # Summary logging
        if cache_hits:
            print(f"[XLA Cache] CACHE HITS ({len(cache_hits)}): {[f'{name} (idx={idx})' for idx, name in cache_hits]}")
        
        if new_inputs:
            print(f"[XLA Cache] CACHE MISSES ({len(new_inputs)}): {[f'{name} (idx={idx})' for idx, name in new_inputs]}")
        
        if static_cache_tensors:
            print(f"[XLA Cache] STATIC CACHE TENSORS FOUND ({len(static_cache_tensors)}):")
            for idx, tensor, cache_key, name in static_cache_tensors:
                hit_or_miss = "HIT" if cache_key in XLAExecutor.previously_seen_inputs else "MISS"
                print(f"[XLA Cache]   {name} -> {hit_or_miss} (tensor_id={id(tensor)}, xla_id={torch_xla._XLAC._xla_get_tensor_id(tensor)})")
        
        print(f"[XLA Cache] Updated cache size: {len(XLAExecutor.previously_seen_inputs)} unique tensors")
        print(f"[XLA Cache] === END CACHE ANALYSIS ===\n")

    def push_tensors_to_device(self, inputs, device):
        if hasattr(inputs, "to"):
            if device not in [inputs.device, inputs.device.type]:
                return inputs.to(device)
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
            print("found key cache or value cache.")
            if hasattr(inputs, "key_cache"):
                inputs.key_cache = self.push_tensors_to_device(inputs.key_cache, device)
            if hasattr(inputs, "value_cache"):
                inputs.value_cache = self.push_tensors_to_device(
                    inputs.value_cache, device
                )
            return inputs
        else:
            return inputs

    def generate_arg_type_map_str(self, output_object):
        
        hlo_input_ids, _ = torch_xla._XLAC._get_tensors_xla_device_data_node(
            output_object
        )

        # xm.get_stablehlo(output_object) gives a graph with just as many inputs as in hlo_input_ids
        print("Hlo input positions pre normalization",hlo_input_ids)
        hlo_input_positions = [id - min(hlo_input_ids) for id in hlo_input_ids]
        print("Hlo input positions post normalization",hlo_input_positions)

        def get_kind_str(kind):
            if kind == InputKind.USER_INPUT:
                return "input"
            elif kind == InputKind.PARAMETER:
                return "parameter"
            else:
                return "constant"

        arg_types = []
        arg_refs = []
        output_args = [o.arg for o in self.program.graph_signature.output_specs]
        
        # debugging - pull out matching target / state dict key
        for in_spec in self.program.graph_signature.input_specs:
            if in_spec.target in self.program.state_dict:
                print("match key in_spec.target", in_spec.target,"with ID", id(self.program.state_dict[in_spec.target]),"and kind", in_spec.kind)
                
        # import pdb; pdb.set_trace()
        
        
        for idx in range(len(hlo_input_positions)):
            if hlo_input_positions[idx] < len(self.program.graph_signature.input_specs):
                in_spec = self.program.graph_signature.input_specs[
                    hlo_input_positions[idx]
                ]
                
                # If an input is passed right through to the output, it will not be
                # captured as an argument
                if in_spec.arg in output_args:
                    continue

                # arg_types.append(get_kind_str(in_spec.kind))
                
                # Get object ID for the corresponding tensor
                if (in_spec.kind == InputKind.PARAMETER or in_spec.kind == InputKind.BUFFER) and in_spec.target in self.program.state_dict:
                    # For parameters, get object ID from state_dict
                    tensor_obj = self.program.state_dict[in_spec.target]
                    arg_refs.append(str(id(tensor_obj)))
                elif in_spec.kind == InputKind.USER_INPUT:
                    # For user inputs, we need to look at the runtime inputs
                    # Since we don't have access to them here, use the input spec arg name as identifier
                    arg_refs.append(f"user_input")
                else:
                    # For constants or other types, use a placeholder
                    arg_refs.append(f"unknown")
            else:
                arg_types.append("constant")
                arg_refs.append("constant_unknown")
                

        self.arg_type_map_str = "main=" + ",".join(arg_types)
        self.arg_ref_map_str = "refs=" + ",".join(arg_refs)

    def __call__(self, *args):
        # Check and log previously seen inputs before pushing to device
        
        args = self.push_tensors_to_device(args, "xla")
        
            
        inputs = self.inputs
        for idx in range(len(args)):
            inputs[self.user_input_indices[idx]] = args[idx]
        
        # inputs is of type 
        # import pdb;pdb.set_trace()
        # for input in inputs:
        #     print("Tensor id via xlac:",torch_xla._XLAC._xla_get_tensor_id(input), "with shape", input.shape)
        
        
        self._check_and_log_previously_seen_inputs(inputs)
        
        
        output = self.program.graph_module(*inputs)
        
        self.generate_arg_type_map_str(output)    
        
        os.environ["ARG_REF_MAP"] = self.arg_ref_map_str
        print("[JAMES] setting arg ref map to ", self.arg_ref_map_str, flush=True)
        
        # if self.compiler_config.arg_type_map_override:
        #     if self.arg_type_map_str is None:
        #         self.generate_arg_type_map_str(output)
        #     if os.environ.get("ARG_TYPE_MAP_OVERRIDE") != self.arg_type_map_str:
        #         os.environ["ARG_TYPE_MAP_OVERRIDE"] = self.arg_type_map_str

        # - `parameter_id_tensor_mapping`
        # - `_get_xla_tensors_dot`
        # - `_get_xla_tensors_text`

        # ctx = torch_xla._XLAC.lowering.LoweringContext()
        # ctx.build([output])
        # host_param_id_tensor_mapping = ctx.parameter_id_tensor_mapping()
        # device_param_id_tensor_mapping = ctx._XLAC.device_parameter_id_tensor_mapping()
        # xla_tensors_text = torch_xla._get_xla_tensors_text() # xla 2.8.0 support req'd
        # import pdb; pdb.set_trace()
        # with open("test.dot", 'w') as fi:
        #     fi.write(torch_xla._get_xla_tensors_dot())
        

        xm.mark_step() # explicit compile step
        dump_static_cache:bool = True
        
        if dump_static_cache:
            for i,_input in enumerate(inputs):
                is_static_cache = _input.dim()==4 and _input.shape[2] == 128 and _input.shape[3]==128
                print(f"input {i}: device = {_input.device}, shape {_input.shape}")
                if is_static_cache:
                    # Avoid implicit device transfer during execution - compute after compilation
                    try:
                        mean_val = torch.mean(_input[0,0,:,:], dim=-1)
                        # Find nonzero indices and count for debugging
                        nonzero_indices = torch.nonzero(mean_val, as_tuple=False).squeeze(-1)
                        nonzero_count = len(nonzero_indices)
                        print(f"\tmean along seqlen for static cache @ input idx {i} and shape {_input.shape}:", mean_val)
                        print(f"\t  nonzero count: {nonzero_count}, nonzero indices: {nonzero_indices.tolist()}")
                    except Exception as e:
                        print(f"\tWarning: Could not compute mean for static cache {i}: {e}")
                        print(f"\tmean along seqlen for static cache @ input idx {i} and shape {_input.shape}: <error during computation>")
        
        if self.compiler_config.push_outputs_to_cpu:
            return self.push_tensors_to_device(output, "cpu")
        

                
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
