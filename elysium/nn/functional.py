from ..autograd.grad_nn import *
from collections.abc import Iterable
from itertools import repeat
def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x[:n]
        return tuple(repeat(x, n))
    return parse
pair = _ntuple(2)
def conv2d(x,w,bias=None,stride=1,padding=0,dilation=1,groups=1,padding_mode='zeros'):
    stride,dilation=pair(stride),pair(dilation)
    padding = padding if isinstance(padding,str) else pair(padding)
    return Convolution.apply(x,w,bias=bias,stride=stride,padding=padding,dilation=dilation,groups=groups,padding_mode=padding_mode)
def conv_transpose2d(x,w,bias=None,stride=1,padding=0,output_padding=0,groups=1,dilation=1):
    stride,padding,dilation,output_padding=pair(stride),pair(padding),pair(dilation),pair(output_padding)
    return TransposedConvolution.apply(x,w,bias=bias,stride=stride,padding=padding,dilation=dilation,groups=groups,output_padding=output_padding)
def maxpool2d(x,kernel_size,stride=None,padding=0,dilation=1,ceil_mode=False,return_indices=False):
    kernel_size ,padding,dilation= pair(kernel_size),pair(padding),pair(dilation)
    stride = kernel_size if stride is None else pair(stride)
    return MaxPool2DWithIndices.apply(x,kernel_size,stride=stride,padding=padding,dilation=dilation,ceil_mode=ceil_mode,return_indices=return_indices)
def avg_pool2d(x,kernel_size,stride=None,padding=0,ceil_mode=False,count_include_pad=True,divisor_override=None):
    kernel_size,padding=pair(kernel_size),pair(padding)
    stride = kernel_size if stride is None else pair(stride)
    return AvgPool2d.apply(x,kernel_size,stride=stride,padding=padding,ceil_mode=ceil_mode,count_include_pad=count_include_pad,divisor_override=divisor_override)
def linear(x, weight, bias=None):
    shape=tuple(i for i in range(len((1,) * (x.ndim - weight.ndim))))
    if bias is not None:
        output = x@weight.transpose(1,0).unsqueeze(shape) + bias
    else:
        output = x@(weight.transpose(1,0).unsqueeze(shape))
    return output
def pad(x,padding,mode='constant',value=None)->'Tensor':
    if mode == 'constant':return ConstantPad2d.apply(x,padding,value)
    if mode == 'reflect' :return ReflectionPad2d.apply(x,padding)
    if mode == 'circular':return CircularPad2d.apply(x,padding)
    if mode == 'replicate':return ReplicationPad2d.apply(x,padding)
def embedding(x, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return Embedding.apply(x,weight,padding_idx=padding_idx,max_norm=max_norm,norm_type=norm_type,scale_grad_by_freq=scale_grad_by_freq,sparse=sparse)
def relu(x,inplace=False)->'Tensor':return ReLU.apply(x,inplace=inplace)
def sigmoid(x)->'Tensor':return Sigmoid.apply(x)

