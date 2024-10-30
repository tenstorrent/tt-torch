import torch
from torch import nn
import pytest

import tt_torch
from tt_torch.tools.verify import verify_module

def test_add():
  class Basic(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, x):
      return x + x
    

  verify_module(Basic(), [(256, 256)])

def test_concat_dim0():
  class Basic(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, x, y):
      return torch.cat((x, y), dim = 0)
    
  verify_module(Basic(), [(32, 32), (64, 32)])

def test_concat_dim1():
  class Basic(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, x, y):
      return torch.cat((x, y), dim = 1)
    
  verify_module(Basic(), [(32, 32), (32, 64)])

def test_concat_dim2():
  class Basic(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, x, y):
      return torch.cat((x, y), dim = 2)
    
  verify_module(Basic(), [(32, 32, 32), (32, 32, 64)])

def test_concat_dim3():
  class Basic(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, x, y):
      return torch.cat((x, y), dim = 3)
    
  verify_module(Basic(), [(32, 32, 32, 32),(32, 32, 32, 64)])

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
  pytest.xfail()  
  class Basic(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear_a = nn.Linear(32, 32)

    def forward(self, x):
      x = self.linear_a(x)
      return x

  verify_module(Basic(), [(32, 32)])


def test_relu():
  pytest.xfail()
  class Basic(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, x):
      return torch.relu(x)

  verify_module(Basic(), [(32, 32)])

def test_rsqrt():
  class Basic(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, x):
      return torch.rsqrt(x)
    
  verify_module(Basic(), [(32, 32)], required_atol=3e-2, input_range=(0.1, 1))

def test_sqrt():
  class Basic(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, x):
      return torch.sqrt(x)
    
  verify_module(Basic(), [(32, 32)], required_atol=3e-2, input_range=(0.1, 1))

dim0_cases = []
for begin in torch.arange(10).tolist():
  for end in torch.arange(90, 100).tolist():
    dim0_cases.append((begin, end, 0))

dim1_cases = []
for begin in torch.arange(10).tolist():
  for end in torch.arange(90, 100).tolist():
    dim1_cases.append((begin, end, 1))

dim2_cases = [] 
for begin in torch.arange(0, 64, 32).tolist():
  for end in torch.arange(64, 128, 32).tolist():
    dim2_cases.append((begin, end, 2))

dim3_cases = []
for begin in torch.arange(0, 64, 32).tolist():
  for end in torch.arange(64, 128, 32).tolist():
    dim3_cases.append((begin, end, 3))

@pytest.mark.parametrize(
  "begin, end, dim",
  [
    *dim2_cases,
    *dim3_cases,
    *dim0_cases,
    *dim1_cases
  ]
)
def test_slice(begin, end, dim):
  class Basic(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, a):
      if dim == 0:
        return a[begin:end, :, :, :]
      elif dim == 1:
        return a[:, begin:end, :, :]
      elif dim == 2:
        return a[:, :, begin:end, :]
      else:
        return a[:, :, :, begin:end]

  shape = [10, 10, 10, 10]
  shape[dim] = 128
  verify_module(Basic(), [shape])

def test_bert():
  pytest.xfail()  
  from torch_mlir import fx
  from torch_mlir.compiler_utils import OutputType
  from transformers import BertModel
  bert = BertModel.from_pretrained("prajjwal1/bert-tiny")
  verify_module(bert, [(1, 32)], input_data_types=[torch.int32])
  
