import math
import os

# import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.autograd import Function
from torch.nn import init, Module, functional
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from typing import Optional, Tuple, List, Union


def hard_sigmoid(x: Tensor):
    return torch.clip((x + 1) / 2, 0, 1)

class Binarize(Function):
    @staticmethod
    def forward(ctx, weight: Tensor, H: float, deterministic: bool=True) -> Tensor:
        weight_binary = hard_sigmoid(weight / H)

        if deterministic:
            weight_binary = torch.round(weight_binary)
        else:
            weight_binary = torch.bernoulli(weight_binary)
            weight_binary = weight_binary.float()

        weight_binary = ((2 * weight_binary - 1) * H).float()
        return weight_binary
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple:
        # grad_output = doutput/dWb
        return grad_output, None, None

binarize = Binarize.apply

class BinaryDense(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
    binary_weight: Tensor

    def __init__(self, in_features: int, out_features: int, H: float=1, bias: bool=False, deterministic: bool=True) -> None:
        super(BinaryDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(Tensor(out_features, in_features))
        self.H = H
        if bias:
            self.bias = Parameter(Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.deterministic = deterministic
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
            
    def forward(self, input: Tensor) -> Tensor:
        weight_binary = binarize(self.weight, self.H, self.deterministic)
        return functional.linear(input, weight_binary, self.bias)        

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class _BinaryConvNd(Module):
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...

    _in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]


    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 H: float=1.,
                 deterministic: bool=True) -> None:
        super(_BinaryConvNd, self).__init__()
        
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        self.H = H
        self.deterministic = deterministic
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_BinaryConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class BinaryConv2D(_BinaryConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        H: float=1.,
        deterministic: bool=True
    ):

        kernel_size_ = torch.nn.modules.utils._pair(kernel_size)
        stride_ = torch.nn.modules.utils._pair(stride)
        padding_ = torch.nn.modules.utils._pair(padding)
        dilation_ = torch.nn.modules.utils._pair(dilation)
        self.H = H
        self.deterministic = deterministic
        super(BinaryConv2D, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return functional.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return functional.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        weight_binary = binarize(self.weight, self.H, self.deterministic)
        return self._conv_forward(input, weight_binary, self.bias)

    
def SquareHingeLoss(input, target):
    # From https://forums.fast.ai/t/custom-loss-function/8647/2
    zero = torch.Tensor([0]).cuda()
    return torch.mean(torch.max(zero, 0.5 - input * target) ** 2)
