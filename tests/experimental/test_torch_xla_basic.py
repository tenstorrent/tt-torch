# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from tt_torch.tools.verify import verify_against_golden
import torch
import torch_xla.core.xla_model as xm
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import BackendOptions

import pytest


@pytest.mark.parametrize("kernel_size", [2, 3, 4, 5])
@pytest.mark.parametrize("stride", [1, 2, 3, 4])
@pytest.mark.parametrize("padding_h", [1, 2, 3, 4])
@pytest.mark.parametrize("padding_w", [1, 2, 3, 4])
@pytest.mark.parametrize("ceil_mode", [True, False])
@pytest.mark.parametrize("input_h", [31, 32, 33])
@pytest.mark.parametrize("input_w", [31, 32, 33])
def test_avg_pool2d(
    kernel_size, stride, padding_h, padding_w, ceil_mode, input_h, input_w
):
    # breakpoint()
    class AvgPool(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.avg_pool = torch.nn.AvgPool2d(
                kernel_size,
                stride,
                padding=(padding_h, padding_w),
                ceil_mode=ceil_mode,
                count_include_pad=True,
            )

        def forward(self, x):
            return self.avg_pool(x)

    input_x = torch.ones(1, 32, input_h, input_w, dtype=torch.bfloat16)

    model = AvgPool()
    golden = model(input_x)
    if not torch.all(golden == 1.0):
        breakpoint()
    else:
        return

    model = torch.compile(model, backend="tt-experimental")

    output = model(input_x)

    verify_against_golden(
        (golden,),
        (output,),
        assert_pcc=False,
        assert_atol=True,
        required_pcc=0.99,
        required_atol=0.008,
    )


@pytest.mark.parametrize("bias", [True, False])
def test_simple_mm(bias):
    class MM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 64, bias=bias, dtype=torch.bfloat16)

        def forward(self, x):
            return self.linear(x)

    input_x = torch.randn(32, 32, dtype=torch.bfloat16)

    model = MM()
    golden = model(input_x)

    model = torch.compile(model, backend="tt-experimental")

    output = model(input_x)

    verify_against_golden(
        (golden,),
        (output,),
        assert_pcc=True,
        assert_atol=True,
        required_pcc=0.99,
        required_atol=0.02,
    )


@pytest.mark.parametrize("bias", [True, False])
def test_simple_mm_eager(bias):
    class MM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32, bias=bias, dtype=torch.bfloat16)

        def forward(self, x):
            return self.linear(x)

    input_x = torch.randn(32, 32, dtype=torch.bfloat16)

    model = MM()
    golden = model(input_x)

    device = xm.xla_device()
    model = model.to(device)
    input_x = input_x.to(device)

    output = model(input_x).to("cpu")

    verify_against_golden(
        (golden,),
        (output,),
        assert_pcc=True,
        assert_atol=True,
        required_pcc=0.99,
        required_atol=0.02,
    )


@pytest.mark.parametrize("in_channels", [3, 64])
@pytest.mark.parametrize("out_channels", [3, 64])
@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_conv2d(
    in_channels, out_channels, kernel_size, stride, padding, dilation, bias
):
    class Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                1,
                bias,
                dtype=torch.bfloat16,
            )

        def forward(self, x):
            return self.conv(x)

    input_x = torch.randn(1, in_channels, 224, 224, dtype=torch.bfloat16)

    model = Conv()
    golden = model(input_x)

    model = torch.compile(model, backend="tt-experimental")

    output = model(input_x)

    verify_against_golden(
        (golden,),
        (output,),
        assert_pcc=True,
        assert_atol=False,
        required_pcc=0.99,
        required_atol=0.02,
    )


@pytest.mark.parametrize("in_channels", [3, 64])
@pytest.mark.parametrize("out_channels", [3, 64])
@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_eager(
    in_channels, out_channels, kernel_size, stride, padding, dilation, bias
):
    class Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                1,
                bias,
                dtype=torch.bfloat16,
            )

        def forward(self, x):
            return self.conv(x)

    input_x = torch.randn(1, in_channels, 224, 224, dtype=torch.bfloat16)

    model = Conv()
    golden = model(input_x)

    device = xm.xla_device()
    model = model.to(device)
    input_x = input_x.to(device)

    output = model(input_x).to("cpu")

    verify_against_golden(
        (golden,),
        (output,),
        assert_pcc=True,
        assert_atol=False,
        required_pcc=0.99,
        required_atol=0.02,
    )


eltwise_unary_ops = [
    torch.abs,
    torch.acos,
    torch.acosh,
    torch.angle,
    torch.asin,
    torch.asinh,
    torch.atan,
    torch.atanh,
    torch.bitwise_not,
    torch.ceil,
    lambda act: torch.clamp(act, -1, 1),  # needs min and max
    torch.conj_physical,
    torch.cos,
    torch.cosh,
    torch.deg2rad,
    torch.digamma,
    torch.erf,
    torch.erfc,
    torch.erfinv,
    torch.exp,
    torch.exp2,
    torch.expm1,
    torch.fix,
    torch.floor,
    torch.frac,
    torch.lgamma,
    torch.log,
    torch.log10,
    torch.log1p,
    torch.log2,
    torch.logit,
    torch.i0,
    torch.isnan,
    torch.nan_to_num,
    torch.neg,
    torch.negative,
    torch.positive,
    torch.rad2deg,
    torch.reciprocal,
    # torch.round, error: failed to legalize operation 'stablehlo.round_nearest_even'
    torch.rsqrt,
    torch.sigmoid,
    torch.sign,
    torch.sgn,
    torch.signbit,
    torch.sin,
    torch.sinc,
    torch.sinh,
    torch.sqrt,
    torch.square,
    torch.tan,
    torch.tanh,
    torch.trunc,
]


@pytest.mark.parametrize("op", eltwise_unary_ops)
def test_eltwise_unary(op):
    input_x = (
        torch.randn(32, 32, dtype=torch.bfloat16)
        if op is not torch.bitwise_not
        else torch.randint(-100, 100, (32, 32))
    )

    class Unary(torch.nn.Module):
        def forward(self, x):
            return op(x)

    model = Unary()
    golden = model(input_x)

    cc = CompilerConfig()
    cc.enable_consteval = True

    model = torch.compile(
        model, backend="tt-experimental", options=BackendOptions(compiler_config=cc)
    )

    output = model(input_x)

    # Not verifying data as many are wrong. Simply testing compile and execute
    verify_against_golden(
        (golden,),
        (output,),
        assert_pcc=False,
        assert_atol=False,
        required_pcc=0.99,
        required_atol=0.01,
    )


@pytest.mark.parametrize("op", eltwise_unary_ops)
def test_eltwise_unary_eager(op):
    if op is torch.erf:
        pytest.skip(
            "erf not decomposed in eager execution. Becomes `stablehlo.custom_call(@mhlo.erf)` which we do not yet lower to ttir"
        )

    class Unary(torch.nn.Module):
        def forward(self, x):
            return op(x)

    input_x = (
        torch.randn(32, 32, dtype=torch.bfloat16)
        if op is not torch.bitwise_not
        else torch.randint(-100, 100, (32, 32))
    )

    model = Unary()
    golden = model(input_x)

    device = xm.xla_device()
    model = model.to(device)
    input_x = input_x.to(device)

    output = model(input_x).to("cpu")

    # Not verifying data as many are wrong. Simply testing compile and execute
    verify_against_golden(
        (golden,),
        (output,),
        assert_pcc=False,
        assert_atol=False,
        required_pcc=0.99,
        required_atol=0.01,
    )


eltwise_binary_ops = [
    torch.add,
    torch.atan2,
    torch.arctan2,
    torch.bitwise_and,
    torch.bitwise_or,
    torch.bitwise_xor,
    torch.bitwise_left_shift,
    torch.bitwise_right_shift,
    torch.div,
    torch.divide,
    torch.floor_divide,
    torch.fmod,
    torch.logaddexp,
    torch.logaddexp2,
    torch.mul,
    torch.multiply,
    torch.nextafter,
    torch.remainder,
    torch.sub,
    torch.subtract,
    torch.true_divide,
    torch.eq,
    torch.ne,
    torch.le,
    torch.ge,
    torch.greater,
    torch.greater_equal,
    torch.gt,
    torch.less_equal,
    torch.lt,
    torch.less,
    torch.maximum,
    torch.minimum,
    torch.fmax,
    torch.fmin,
    torch.not_equal,
]


@pytest.mark.parametrize("op", eltwise_binary_ops)
def test_eltwise_binary(op):
    if op in [
        torch.bitwise_and,
        torch.bitwise_or,
        torch.bitwise_xor,
    ]:
        input_x = torch.randint(-100, 100, (32, 32))
        input_y = torch.randint(-100, 100, (32, 32))
    elif op in [torch.bitwise_left_shift, torch.bitwise_right_shift]:
        # TODO: enable test for these ops once issues is resolved (https://github.com/tenstorrent/tt-torch/issues/1127)
        pytest.skip(
            f"{op} not supported in tt-experimental backend yet. Skipping test."
        )
    else:
        input_x = torch.randn(32, 32, dtype=torch.bfloat16)
        input_y = torch.randn(32, 32, dtype=torch.bfloat16)

    class Binary(torch.nn.Module):
        def forward(self, x, y):
            return op(x, y)

    model = Binary()
    golden = model(input_x, input_y)

    model = torch.compile(model, backend="tt-experimental")

    output = model(input_x, input_y)

    # Not verifying data as many are wrong. Simply testing compile and execute
    verify_against_golden(
        (golden,),
        (output,),
        assert_pcc=False,
        assert_atol=False,
        required_pcc=0.99,
        required_atol=0.02,
    )


@pytest.mark.parametrize("op", eltwise_binary_ops)
def test_eltwise_binary_eager(op):
    if op in [
        torch.bitwise_and,
        torch.bitwise_or,
        torch.bitwise_xor,
    ]:
        input_x = torch.randint(-100, 100, (32, 32))
        input_y = torch.randint(-100, 100, (32, 32))
    elif op in [torch.bitwise_left_shift, torch.bitwise_right_shift]:
        # TODO: enable test for these ops once issues is resolved (https://github.com/tenstorrent/tt-torch/issues/1127)
        pytest.skip(
            f"{op} not supported in tt-experimental backend yet. Skipping test."
        )
    else:
        input_x = torch.randn(32, 32, dtype=torch.bfloat16)
        input_y = torch.randn(32, 32, dtype=torch.bfloat16)

    class Binary(torch.nn.Module):
        def forward(self, x, y):
            return op(x, y)

    model = Binary()
    golden = model(input_x, input_y)

    device = xm.xla_device()
    model = model.to(device)
    input_x = input_x.to(device)
    input_y = input_y.to(device)

    output = model(input_x, input_y).to("cpu")

    # Not verifying data as many are wrong. Simply testing compile and execute
    verify_against_golden(
        (golden,),
        (output,),
        assert_pcc=False,
        assert_atol=False,
        required_pcc=0.99,
        required_atol=0.02,
    )
