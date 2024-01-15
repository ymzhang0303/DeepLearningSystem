"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        
        self.padding = (kernel_size - 1) // 2
        self.weight = Parameter(init.kaiming_uniform(in_channels * kernel_size *kernel_size, 
                                                    out_channels * kernel_size *kernel_size, 
                                                    (kernel_size, kernel_size, in_channels, out_channels), 
                                                    device=device, dtype=dtype))
        self.use_bias = bias
        if self.use_bias:
            bias_bound = 1.0 / (self.in_channels * self.kernel_size ** 2) ** 0.5
            self.bias = Parameter(init.rand(self.out_channels, low=-bias_bound, high=bias_bound, device=device))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        nhwc = x.transpose((1, 2)).transpose((2,3))
        out_nhwc = ops.conv(nhwc, self.weight, stride=self.stride, padding=self.padding)
        if self.use_bias:
            out_nhwc += self.bias.reshape((1,1,1,self.out_channels)).broadcast_to(out_nhwc.shape)
        return out_nhwc.transpose((2, 3)).transpose((1, 2))
        ### END YOUR SOLUTION