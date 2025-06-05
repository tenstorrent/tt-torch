# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest
from torch._dynamo import optimize
from torch._functorch.aot_autograd import aot_module_simplified
from tt_torch.dynamo.backend import backend
from tt_torch.tools.utils import (
    CompilerConfig,
)
import copy


def test_training():
    def my_fw_compiler(fx_graph, example_inputs):
        return torch.compile(fx_graph, backend=backend, dynamic=False)

    def my_bw_compiler(fx_graph, example_inputs):
        return torch.compile(fx_graph, backend=backend, dynamic=False)

    def cpu_fw_compiler(fx_graph, example_inputs):
        return fx_graph.forward

    def cpu_bw_compiler(fx_graph, example_inputs):
        return fx_graph.forward

    class TwoMatmulWithParams(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Parameter(
                torch.clamp(torch.randn(8, 16, requires_grad=True), min=-1.0, max=1.0)
            )
            self.w2 = nn.Parameter(
                torch.clamp(torch.randn(16, 32, requires_grad=True), min=-1.0, max=1.0)
            )

        def forward(self, x):
            out1 = torch.matmul(x, self.w1)
            out2 = torch.matmul(out1, self.w2)
            return (out2,)

    x = torch.clamp(torch.randn(4, 8, requires_grad=False), max=1.0)
    w1 = torch.randn(8, 16, requires_grad=True)
    w2 = torch.randn(16, 32, requires_grad=True)
    # model = TwoMatmul()
    model = TwoMatmulWithParams()
    model_cpu = copy.deepcopy(model)

    compiled_forward_backward_fn = aot_module_simplified(
        model, (x,), fw_compiler=my_fw_compiler, bw_compiler=my_bw_compiler
    )

    compiled_forward_backward_fn_cpu = aot_module_simplified(
        model_cpu, (x,), fw_compiler=cpu_fw_compiler, bw_compiler=cpu_bw_compiler
    )

    forward_output = compiled_forward_backward_fn(
        x,
    )[0]
    forward_output_cpu = compiled_forward_backward_fn_cpu(
        x,
    )[0]
    result = torch.allclose(forward_output, forward_output_cpu, atol=1e-01, rtol=1e-01)

    target = torch.randn_like(forward_output)
    loss_fn = nn.MSELoss()
    loss = loss_fn(forward_output, target)
    print(f"\nLoss: {loss.item()}")
    loss.backward()

    loss_cpu = loss_fn(forward_output_cpu, target)
    print(f"\nLoss CPU: {loss.item()}")
    loss_cpu.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name}:")
            print(param.grad)
        else:
            print(f"No gradient for {name}")
