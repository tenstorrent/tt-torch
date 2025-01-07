# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/linear_autoencoder/pytorch_linear_autoencoder.py

import torch
import torchvision.transforms as transforms
from tt_torch.tools.verify import verify_module
from tt_torch.tools.utils import CompilerConfig, CompileDepth

def test_autoencoder():
    class AutoEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

            # Encoder
            self.encoder_lin1 = torch.nn.Linear(784, 128)
            self.encoder_lin2 = torch.nn.Linear(128, 64)
            self.encoder_lin3 = torch.nn.Linear(64, 12)
            self.encoder_lin4 = torch.nn.Linear(12, 3)

            # Decoder
            self.decoder_lin1 = torch.nn.Linear(3, 12)
            self.decoder_lin2 = torch.nn.Linear(12, 64)
            self.decoder_lin3 = torch.nn.Linear(64, 128)
            self.decoder_lin4 = torch.nn.Linear(128, 784)

            # Activation Function
            self.act_fun = torch.nn.ReLU()

        def forward(self, x):
            # Encode
            act = self.encoder_lin1(x)
            act = self.act_fun(act)
            act = self.encoder_lin2(act)
            act = self.act_fun(act)
            act = self.encoder_lin3(act)
            act = self.act_fun(act)
            act = self.encoder_lin4(act)

            # Decode
            act = self.decoder_lin1(act)
            act = self.act_fun(act)
            act = self.decoder_lin2(act)
            act = self.act_fun(act)
            act = self.decoder_lin3(act)
            act = self.act_fun(act)
            act = self.decoder_lin4(act)

            return act

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.remove_embedded_constants = True

    verify_module(AutoEncoder(), input_shapes=[(1, 784)], compiler_config=cc)    

def test_encoder_linear_one():
    class EncoderLinearOne(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder_lin1 = torch.nn.Linear(784, 128)

        def forward(self, x):
            act = self.encoder_lin1(x)
            return act

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.remove_embedded_constants = True

    verify_module(EncoderLinearOne(), input_shapes=[(1, 784)], compiler_config=cc)

def test_encoder_linear_two():
    class EncoderLinearTwo(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder_lin1 = torch.nn.Linear(128, 64)

        def forward(self, x):
            act = self.encoder_lin1(x)
            return act

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.remove_embedded_constants = True

    verify_module(EncoderLinearTwo(), input_shapes=[(1, 128)], compiler_config=cc)

def test_encoder_linear_three():
    class EncoderLinearThree(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder_lin1 = torch.nn.Linear(64, 12)

        def forward(self, x):
            act = self.encoder_lin1(x)
            return act

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.remove_embedded_constants = True

    verify_module(EncoderLinearThree(), input_shapes=[(1, 64)], compiler_config=cc)

def test_encoder_linear_four():
    class EncoderLinearFour(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder_lin1 = torch.nn.Linear(12, 3)

        def forward(self, x):
            act = self.encoder_lin1(x)
            return act

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.remove_embedded_constants = True

    verify_module(EncoderLinearFour(), input_shapes=[(1, 12)], compiler_config=cc)

def test_decoder_linear_one():
    class DecoderLinearOne(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.decoder_lin1 = torch.nn.Linear(3, 12)

        def forward(self, x):
            act = self.decoder_lin1(x)
            return act

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.remove_embedded_constants = True

    verify_module(DecoderLinearOne(), input_shapes=[(1, 3)], compiler_config=cc)

def test_decoder_linear_two():
    class DecoderLinearTwo(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.decoder_lin1 = torch.nn.Linear(12, 64)

        def forward(self, x):
            act = self.decoder_lin1(x)
            return act

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.remove_embedded_constants = True

    verify_module(DecoderLinearTwo(), input_shapes=[(1, 12)], compiler_config=cc)

def test_decoder_linear_three():
    class DecoderLinearThree(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.decoder_lin1 = torch.nn.Linear(64, 128)

        def forward(self, x):
            act = self.decoder_lin1(x)
            return act

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.remove_embedded_constants = True

    verify_module(DecoderLinearThree(), input_shapes=[(1, 64)], compiler_config=cc)

def test_decoder_linear_four():
    class DecoderLinearFour(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.decoder_lin1 = torch.nn.Linear(128, 784)

        def forward(self, x):
            act = self.decoder_lin1(x)
            return act

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.remove_embedded_constants = True

    verify_module(DecoderLinearFour(), input_shapes=[(1, 128)], compiler_config=cc)

def test_relu_one():
    class Relu(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.act_fun = torch.nn.ReLU()

        def forward(self, x):
            act = self.act_fun(x)
            return act

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.remove_embedded_constants = True

    verify_module(Relu(), input_shapes=[(1, 12)], compiler_config=cc)

def test_relu_two():
    class Relu(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.act_fun = torch.nn.ReLU()

        def forward(self, x):
            act = self.act_fun(x)
            return act

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.remove_embedded_constants = True

    verify_module(Relu(), input_shapes=[(1, 64)], compiler_config=cc)

def test_relu_three():
    class Relu(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.act_fun = torch.nn.ReLU()

        def forward(self, x):
            act = self.act_fun(x)
            return act

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    cc.remove_embedded_constants = True

    verify_module(Relu(), input_shapes=[(1, 128)], compiler_config=cc)

test_autoencoder()
