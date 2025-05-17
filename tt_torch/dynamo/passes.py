# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.fx.experimental import const_fold
from typing import List, Optional, Union
from torch.export.graph_signature import InputKind
from tt_torch.tools.utils import RuntimeIntermediate

from tt_torch.tools.utils import MultiChipInput, MultiChipOutput, IOType, MultiChipGraph

from .decompositions import (
    CUSTOM_DECOMPOSITION_TABLE,
)


def run_shape_prop(gm, example_inputs):
    shape_prop = torch.fx.passes.shape_prop.ShapeProp(gm)
    if shape_prop.fake_mode is not None:
        fake_args = [
            shape_prop.fake_mode.from_tensor(act, static_shapes=True)
            if isinstance(act, torch.Tensor)
            else act
            for act in example_inputs
        ]
    else:
        fake_args = example_inputs
    shape_prop.run(*fake_args)


def reduce_graph(module_or_graph: Union[torch.fx.Graph, torch.fx.GraphModule]):
    # Reduce the graph to only the nodes that are used

    # Traverse up the graph from output nodes to populate consumed nodes set
    graph = (
        module_or_graph.graph
        if isinstance(module_or_graph, torch.fx.GraphModule)
        else module_or_graph
    )
    consumed = set()
    working_nodes = []
    for node in graph.nodes:
        if node.op == "output":
            working_nodes.append(node)
            consumed.add(node)

    while len(working_nodes) > 0:
        node = working_nodes.pop(0)
        if not isinstance(node, torch.fx.Node):
            continue
        for arg in node.all_input_nodes:
            if arg not in consumed:
                consumed.add(arg)
                working_nodes.append(arg)

    for node in reversed(graph.nodes):
        if node not in consumed:
            graph.erase_node(node)


def bypass_redundant_getitem(gm):
    for node in gm.graph.nodes:
        if node.op == "call_function" and "getitem" in node.name:
            if isinstance(node.args[0], tuple):
                idx = node.args[1]
                if isinstance(idx, int):
                    node.replace_all_uses_with(node.args[0][idx])
    return gm


def bypass_redundant_cast(gm):
    # Removes cast nodes that cast to already existing dtype
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and hasattr(node.target, "name")
            and "prims::convert_element_type" in node.target.name()
        ):
            if node.args[1] == node.args[0].meta["tensor_meta"].dtype:
                node.replace_all_uses_with(node.args[0])

    return gm


def bypass_dtype_promotion(gm, compiler_config):
    # Removes casting of nodes to float32 unless they were explicitly cast by the user.
    # Pytorch insists on casting params to float32, even though the user may have specified a different dtype,
    # and forcing certain decomposition (i.e. adaptive_avg_pool2d) to be in float32
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and hasattr(node.target, "name")
            and "prims::convert_element_type" in node.target.name()
        ):
            if (
                node.meta["original_aten"]._name != "aten::_to_copy"
                and node.args[1] == torch.float32
            ):
                node.replace_all_uses_with(node.args[0])

    return gm


def constant_fold(gm):
    gm = const_fold.split_const_subgraphs(gm)
    gm.run_folding()

    gm.graph.eliminate_dead_code()
    return gm


def node_to_device(node, device_map):
    if (
        not hasattr(node, "meta")
        or "nn_module_stack" not in node.meta
        or len(node.meta["nn_module_stack"]) == 0
    ):
        return None

    # The last stack contains the most information, only relevent fields will be used
    # Contains string like: "L['self']._modules['model']._modules['layers']._modules['30'].mlp.up_proj"
    # or like "L['self'].model.embed_tokens"
    module_stack = list(node.meta["nn_module_stack"].values())[-1][0]

    vals = module_stack.rsplit(".")[1:]
    parsed_vals = []
    for val in vals:
        if val.startswith("_modules['"):
            parsed_vals.append(val[10:-2])
        else:
            parsed_vals.append(val)

    # append layers to each other until we find something in the device map. This needs to be done because
    # the model can be split at model.layers.1 or model.layers.1.mlp
    for i in range(1, len(parsed_vals) + 1):
        layer = ".".join(parsed_vals[:i])
        if layer in device_map:
            return device_map[layer]

    print(f"Warning: No device found for node {node}")
    return None


def flatten_args(args):
    flattened = []

    def _flatten(obj):
        if isinstance(obj, list):
            return ["list", [_flatten(x) for x in obj]]
        elif isinstance(obj, tuple):
            return ["tuple", [_flatten(x) for x in obj]]
        elif isinstance(obj, dict):
            return ["dict", {k: _flatten(v) for k, v in obj.items()}]
        else:
            flattened.append(obj)
            return None  # leaf node

    structure = _flatten(args)
    return flattened, structure


def rebuild_args(flattened, structure):
    flat_iter = iter(flattened)

    def _rebuild(struct):
        if struct is None:
            return next(flat_iter)
        kind, content = struct
        if kind == "list":
            return [_rebuild(x) for x in content]
        elif kind == "tuple":
            return tuple(_rebuild(x) for x in content)
        elif kind == "dict":
            return {k: _rebuild(v) for k, v in content.items()}
        else:
            raise TypeError(f"Unknown structure kind: {kind}")

    return _rebuild(structure)


def dump_graph(graph, file_name):
    with open(file_name, "w") as f:
        f.write(str(graph))


def _generate_golden_intermediate_cache(gm, inputs, compiler_config):
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
            compiler_config.runtime_intermediate_cache[node.name] = cache_entry
            print(f"Caching runtime intermediate for {node.name}")
            tensor = node.target(*args, **node.kwargs)
            node_to_tensor[node] = tensor


# The following function splits the graph onto the devices specified in the device_map
# We create empty subgraphs for each device and then add the nodes to the appropriate subgraph
def split_onto_devices(gm, compiler_config):
    device_indices = set(compiler_config.device_map.values())
    if len(device_indices) == 0:
        device_indices = [0]
    mcg = MultiChipGraph(device_indices)
    if len(device_indices) == 1:
        mcg.device_graphs = {0: gm.graph}
        input_index = 0
        output_index = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                mci = MultiChipInput(
                    0,
                    IOType.USER,
                    input_index,
                    input_index,
                )
                mcg.graph_inputs[0].append(mci)
                input_index += 1
            elif node.op == "output":
                for _ in node.args[0]:
                    mco = MultiChipOutput(0, IOType.USER, output_index)
                    mcg.graph_outputs[0].append(mco)
                    output_index += 1
        return mcg

    node_to_new_nodes = {}
    user_input_index = 0
    consumer_input_indices = [0] * len(device_indices)
    output_indices = [0] * len(device_indices)
    outputs = [[] for _ in device_indices]
    prev_device_idx = None

    def update_device_index(node):
        nonlocal prev_device_idx
        device_idx = node_to_device(node, compiler_config.device_map)
        if device_idx is None:
            assert prev_device_idx is not None
            device_idx = prev_device_idx
        prev_device_idx = device_idx
        return device_idx

    for node in gm.graph.nodes:
        if node.op == "get_attr" or node.op == "placeholder":
            is_placeholder = node.op == "placeholder"
            user_devices = [
                node_to_device(user, compiler_config.device_map) for user in node.users
            ]
            user_devices = [d for d in user_devices if d is not None]
            devices = list(set(user_devices))
            if len(devices) == 0:
                assert prev_device_idx is not None
                devices = [prev_device_idx]
            for device_idx in devices:
                prev_device_idx = device_idx
                inp_node = (
                    mcg.device_graphs[device_idx].placeholder(node.target)
                    if is_placeholder
                    else mcg.device_graphs[device_idx].get_attr(node.target)
                )
                inp_node.meta = node.meta
                node_to_new_nodes[node] = {device_idx: inp_node}
                if is_placeholder:
                    mci = MultiChipInput(
                        device_idx,
                        IOType.USER,
                        user_input_index,
                        consumer_input_indices[device_idx],
                    )
                    mcg.graph_inputs[device_idx].append(mci)
                    consumer_input_indices[device_idx] += 1
                    user_input_index += 1
            # TODO Assert on graphs that feed each other

        elif (
            node.op == "call_function"
            or node.op == "call_method"
            or node.op == "call_module"
        ):
            device_idx = update_device_index(node)
            graph = mcg.device_graphs[device_idx]
            node_args = []
            node_kw_args = []
            flattened_args, structure_args = flatten_args(node.args)
            flattened_kwargs, structure_kwargs = flatten_args(node.kwargs)

            def _process_arg(arg):
                if isinstance(arg, torch.fx.node.Node):
                    if arg in node_to_new_nodes:
                        new_arg = node_to_new_nodes[arg]
                        if device_idx in new_arg:
                            return new_arg[device_idx]
                        else:
                            feeding_device = list(new_arg.keys())[0]
                            outputs[feeding_device].append(new_arg[feeding_device])
                            mco = MultiChipOutput(
                                feeding_device,
                                IOType.INTER_DEVICE,
                                output_indices[feeding_device],
                            )
                            mcg.graph_outputs[feeding_device].append(mco)
                            # TODO Assert on graphs that feed each other
                            placeholder = graph.placeholder(
                                new_arg[feeding_device].name
                            )
                            placeholder.meta = new_arg[feeding_device].meta
                            node_to_new_nodes[arg][device_idx] = placeholder
                            mci = MultiChipInput(
                                device_idx,
                                IOType.INTER_DEVICE,
                                output_indices[feeding_device],
                                consumer_input_indices[device_idx],
                            )
                            mco.link_input(mci)
                            mcg.graph_inputs[device_idx].append(mci)
                            consumer_input_indices[device_idx] += 1
                            output_indices[feeding_device] += 1
                            return placeholder
                    else:
                        assert False
                else:
                    return arg

            for arg in flattened_args:
                node_args.append(_process_arg(arg))
            for arg in flattened_kwargs:
                node_kw_args.append(_process_arg(arg))

            if len(node_args) != len(node.args):
                # are any of the args duplicates? If so, we need to duplicate the placeholders
                for idx, arg in enumerate(node.args):
                    if arg in node.args[idx + 1 :]:
                        node_args.append(node_args[idx])

            rebuilt_args = rebuild_args(node_args, structure_args)
            rebuilt_kwargs = rebuild_args(node_kw_args, structure_kwargs)
            if node.op == "call_function":
                node_creator = graph.call_function
            elif node.op == "call_method":
                node_creator = graph.call_method
            elif node.op == "call_module":
                node_creator = graph.call_module
            else:
                assert False
            new_node = node_creator(node.target, tuple(rebuilt_args), rebuilt_kwargs)
            new_node.meta = node.meta
            new_node.name = node.name
            node_to_new_nodes[node] = {device_idx: new_node}

        elif node.op == "output":
            # Final outputs
            final_outputs = node.args[0]
            for index, output in enumerate(final_outputs):
                device_idx = update_device_index(output)
                outputs[device_idx].append(node_to_new_nodes[output][device_idx])
                mco = MultiChipOutput(device_idx, IOType.USER, index)
                mcg.graph_outputs[device_idx].append(mco)
        else:
            assert False

    for idx, output in enumerate(outputs):
        graph = mcg.device_graphs[idx]
        if len(output) == 1:
            output = output[0]
        else:
            output = tuple(output)
        graph.output(output)

    for graph in mcg.device_graphs.values():
        graph.lint()

    return mcg


def prune_inputs(program, constant_inputs):
    placeholder_index = 0
    indices_to_remove = []
    for node in program.graph_module.graph.nodes:
        if node.op == "placeholder":
            if len(node.users) == 0:
                indices_to_remove.append(placeholder_index)
                program.graph_module.graph.erase_node(node)
            placeholder_index += 1
            if placeholder_index == len(constant_inputs):
                break

    program.graph_module.graph.eliminate_dead_code()
    constant_inputs = [
        constant_inputs[i]
        for i in range(len(constant_inputs))
        if i not in indices_to_remove
    ]
    program._graph_signature.input_specs = [
        input_spec
        for i, input_spec in enumerate(program._graph_signature.input_specs)
        if i not in indices_to_remove
    ]
    return constant_inputs


def pass_pipeline(gm: torch.fx.GraphModule, example_inputs, compiler_config):
    decompositions = torch.export.default_decompositions()
    decompositions.update(CUSTOM_DECOMPOSITION_TABLE)
    mcg = split_onto_devices(gm, compiler_config)

    for idx, graph in mcg.device_graphs.items():
        sub_example_inputs = []
        for node in graph.nodes:
            if node.op == "placeholder":
                if "tensor_meta" in node.meta:
                    sub_example_inputs.append(
                        torch.randn(node.meta["tensor_meta"].shape).to(
                            dtype=node.meta["tensor_meta"].dtype
                        )
                    )
                else:
                    assert "example_value" in node.meta
                    sub_example_inputs.append(
                        torch.randn(node.meta["example_value"].shape).to(
                            dtype=node.meta["example_value"].dtype
                        )
                    )

        gm_device = torch.fx.GraphModule(gm, graph, f"_device_{idx}")
        # we use the export API to run the decompositions, as this maintains the
        # source locations in stack_trace
        gm_device = (
            torch.export.export_for_training(
                gm_device, tuple(sub_example_inputs), strict=False
            )
            .run_decompositions(decompositions)
            .module()
        )
        gm_device = bypass_dtype_promotion(gm_device, compiler_config)
        # shape prop also propagates dtypes, need to run to figure out which casts are redundant
        run_shape_prop(gm_device, sub_example_inputs)
        gm_device = bypass_redundant_cast(gm_device)

        if compiler_config.enable_consteval:
            gm_device = constant_fold(gm_device)
        elif compiler_config.consteval_parameters:
            raise Exception(
                "consteval_parameters is enabled but enable_consteval is not"
            )

        gm_device = bypass_redundant_getitem(gm_device)

        # reduce_graph(gm) - ISSUE: https://github.com/tenstorrent/tt-torch/issues/513
        program = torch.export.export(
            gm_device, tuple(sub_example_inputs), strict=False
        )
        # The proper order of inputs when outlining everything is constants + parameters + buffers + sub_example_inputs
        if not compiler_config.inline_parameters:
            constant_inputs = (
                list(program.tensor_constants.values())
                + [
                    param.contiguous() if not param.is_contiguous() else param
                    for param in program.parameters()
                ]
                + list(program.buffers())
            )
            for i in range(len(program._graph_signature.input_specs)):
                if program._graph_signature.input_specs[i].kind != InputKind.USER_INPUT:
                    program._graph_signature.input_specs[i].kind = InputKind.USER_INPUT
        else:
            constant_inputs = []

        constant_inputs = prune_inputs(program, constant_inputs)
        run_shape_prop(program.graph_module, constant_inputs + sub_example_inputs)
        mcg.programs[idx] = program
        mcg.constant_inputs[idx] = constant_inputs
        mcg.example_inputs[idx] = sub_example_inputs

    if compiler_config._enable_intermediate_verification:
        if len(mcg.programs) > 1:
            assert (
                False
            ), "Intermediate verification is not supported for multi-chip models"

        # Once a program is generated, it should not be mutated, so we can safely generate the golden intermediate cache here
        _generate_golden_intermediate_cache(
            mcg.programs[0],
            mcg.constant_inputs[0] + mcg.example_inputs[0],
            compiler_config,
        )

    return mcg
