import torch
from torch import nn
import pytest

from tt_torch.tools.verify import verify_module

def test_add():
  class Basic(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, x):
      return x + x
    

  verify_module(Basic(), [(256, 256)])

def test_linear():
  class Basic(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear_a = nn.Linear(32, 64, bias=False)
      self.linear_b = nn.Linear(64, 64, bias=False)

    def forward(self, x):
      x = self.linear_a(x)
      x = self.linear_b(x)
      return x

  verify_module(Basic(), [(32, 32)])

from torch_mlir import fx
from torch_mlir.compiler_utils import OutputType

def test_linear_with_bias():
  class Basic(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear_a = nn.Linear(32, 32)

    def forward(self, x):
      x = self.linear_a(x)
      return x

  verify_module(Basic(), [(32, 32)])


def test_relu():
  pytest.skip()
  class Basic(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, x):
      return torch.relu(x)

  verify_module(Basic(), [(32, 32)])

def test_bert():
  from torch_mlir import fx
  from torch_mlir.compiler_utils import OutputType
  from transformers import BertModel
  bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
  verify_module(bert, [(1, 32)], input_data_types=[torch.int32])