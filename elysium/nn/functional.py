from elysium import*
from ..autograd.grad_nn import *

def conv2d(x,w,bias=None,stride=1,padding=0,dilation=1,groups=1,padding_mode='zeros'):
    if isinstance(stride,int):stride=(stride,stride)
    if isinstance(padding,int):padding=(padding,padding)
    if isinstance(dilation,int):dilation=(dilation,dilation)
    return Convolution.apply(x,w,bias=bias,stride=stride,padding=padding,dilation=dilation,groups=groups,padding_mode=padding_mode)

