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
def dilate(x,dilation)->'Tensor':return Dilate.apply(x,dilation)
def flip(x,axis=None)->'Tensor':return Flip.apply(x,axis=axis)
def pad(x,padding,stride=None, dilation=None, kernel_size=None,mode='zeros',value=None)->'Tensor':
    if mode == 'zeros':
        if value is None:
            return ConstantPad2d.apply(x,padding,0.0,stride=stride, dilation=dilation, kernel_size=kernel_size)
        return ConstantPad2d.apply(x,padding,value,stride=stride, dilation=dilation, kernel_size=kernel_size)
    if mode == 'reflect' :return ReflectionPad2d.apply(x,padding,stride=stride, dilation=dilation, kernel_size=kernel_size)
    if mode == 'circular':return CircularPad2d.apply(x,padding,stride=stride, dilation=dilation, kernel_size=kernel_size)
    if mode == 'replicate':return ReplicationPad2d.apply(x,padding,stride=stride, dilation=dilation, kernel_size=kernel_size)
def conv2d_impl(x,w,bias,stride,dilation,groups):
    return Convolution.apply(x,w,bias=bias,stride=stride,dilation=dilation,groups=groups)
def conv2d(x,w,bias=None,stride=1,padding=0,dilation=1,groups=1,padding_mode='zeros'):
    kernel_size,stride,dilation=pair(w.shape[-2:]),pair(stride),pair(dilation)
    padding = padding if isinstance(padding,str) else pair(padding)
    if padding != (0,0):
        x = pad(x,padding,stride=stride, dilation=dilation, kernel_size=kernel_size,mode=padding_mode)
        return conv2d_impl(x,w,bias,stride=stride,dilation=dilation,groups=groups)
    return conv2d_impl(x,w,bias,stride,dilation,groups)
def conv_transpose2d(x,w,bias=None,stride=1,padding=0,output_padding=0,groups=1,dilation=1):
    stride,padding,dilation,output_padding=pair(stride),pair(padding),pair(dilation),pair(output_padding)
    ph,pw = padding
    c_in,c_out_group,kh,kw = w.shape
    x_dilated = dilate(x,stride)
    x_padded_dilated = pad(x_dilated,((kh - 1) * dilation[0], (kw - 1) * dilation[1]))
    w = w.reshape((groups,c_in//groups,c_out_group ,kh,kw))
    fliped_w = flip(w,axis=(-1,-2)).transpose(1,2)
    fliped_w = fliped_w.reshape((c_out_group*groups,c_in//groups,fliped_w.shape[3],fliped_w.shape[4]))
    y=conv2d(x_padded_dilated,fliped_w,bias=bias,stride=(1,1),dilation=dilation,groups=groups)
    if output_padding != (0,0):
        hop, wop = output_padding if len(output_padding) == 2 else (output_padding[0], output_padding[0])
        Hx = (x.shape[-2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (w.shape[-2] - 1) + hop + 1
        Wx = (x.shape[-1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (w.shape[-1] - 1) + wop + 1
        return y[...,ph:Hx + ph,pw:Wx+pw]
    return y[...,ph:y.shape[2]-ph,pw:y.shape[3]-pw]

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
        output = x@weight.transpose(1,0).unsqueeze(shape) + bias if len(shape) > 0 else x@weight.transpose(1,0) + bias 
    else:
        output = x@(weight.transpose(1,0).unsqueeze(shape))
    return output
def embedding(x, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return Embedding.apply(x,weight,padding_idx=padding_idx,max_norm=max_norm,norm_type=norm_type,scale_grad_by_freq=scale_grad_by_freq,sparse=sparse)
def relu(x,inplace=False)->'Tensor':return ReLU.apply(x,inplace=inplace)
def sigmoid(x)->'Tensor':return Sigmoid.apply(x)
def logsigmoid(x)->'Tensor':return LogSigmoid.apply(x)
def leaky_relu(x, negative_slope=0.01, inplace=False)->'Tensor':return LeakyReLU.apply(x,negative_slope=negative_slope,inplace=inplace)
def tanh(x)->'Tensor':return Tanh.apply(x)
def softmax(x, dim=None)->'Tensor':return Softmax.apply(x,dim=dim)
def log_softmax(x, dim=None)->'Tensor':return LogSoftmax.apply(x,dim=dim)
def batch_norm_impl(x,running_mean,running_var,weight=None,bias=None,training=False,momentum=0.1,eps=1e-05)->'Tensor':
    if training:
        var = x.var((0,2,3),correction=0)
        mean= x.mean(axis=(0,2,3))
        if running_mean is not None and running_var is not None:
            running_mean = (1 - momentum) * running_mean + mean * momentum 
            running_var  = (1 - momentum) * running_var + x.var((0,2,3),correction=1) * momentum 
        out = (x - mean[None,:,None,None]) / (var[None,:,None,None] + eps).sqrt()
    else:
        if running_mean is not None and running_var is not None:
            mean ,var = running_mean[None,:,None,None],running_var[None,:,None,None]
        else:
            mean,var = x.mean(axis=(0,2,3),keepdim=True),x.var((0,2,3),correction=0,keepdim=True)
        out = (x - mean) / (var + eps).sqrt()
    if weight is not None:out = weight[None,:,None,None] * out + (bias[None,:,None,None] if bias is not None else 0)
    return out,running_mean,running_var
def batch_norm(x,running_mean,running_var,weight=None,bias=None,training=False,momentum=0.1,eps=1e-05):
    return batch_norm_impl(x,running_mean,running_var,weight=weight,bias=bias,training=training,momentum=momentum,eps=eps)[0]
def layer_norm(x,normalized_shape,weight=None,bias=None,eps=1e-05)->'Tensor':
    axis = tuple(ax if ax >= 0 else x.ndim + ax for ax in tuple(range(-len(normalized_shape), 0)))
    mean = x.mean(axis=axis,keepdim=True)
    var = x.var(dim=axis, correction=0,keepdim=True) if weight is not None else x.var(dim=axis, correction=1, keepdim=True)
    x_normalized = (x - mean) / (var + eps).sqrt()
    if weight is not None :
        x_normalized = weight * x_normalized + bias if bias is not None else 0
    return x_normalized
def group_norm(x, num_groups, weight=None, bias=None, eps=1e-05)->'Tensor':
    assert x.shape[1] % num_groups == 0,f'Number of groups must be divisible by input channels'
    N, C, H, W = x.shape
    x=x.reshape((N,num_groups,C // num_groups,H,W))
    mean = x.mean(axis=(2,3,4),keepdim=True)
    var = x.var(dim=(2,3,4),correction=0,keepdim=True)
    out = (x-mean) / (var + eps).sqrt()
    out=out.reshape((N,C,H,W))
    if weight is not None:
        out = weight[None,:,None,None] * out + (bias[None,:,None,None] if bias is not None else 0)
    return out
def instance_norm(x,running_mean=None,running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-05):
    if use_input_stats:
        mean = x.mean((2,3),keepdim=True)
        var = x.var((2,3),correction=0,keepdim=True)
        if running_mean is not None and running_var is not None:
            running_mean = (1 - momentum) * running_mean + x.mean((0,2,3)) * momentum
            running_var  = (1 - momentum) * running_var + x.var(dim=(0,2,3),correction=1) * momentum
        out = (x - mean) / (var + eps).sqrt()
    else:
        if running_mean is None or running_var is None:
            raise ValueError("Running mean and variance must be provided when use_input_stats=False")
        out = (x - running_mean[None,:,None,None]) / (running_var[None,:,None,None] + eps).sqrt()
    if weight is not None:
        out = weight[None,:,None,None] * out + (bias[None,:,None,None] if bias is not None else 0)
    return out
def l1_loss(x:'Tensor', target:'Tensor',reduction='mean')->'Tensor':return L1Loss.apply(x,target,reduction=reduction)
def mse_loss(x:'Tensor',target:'Tensor',reduction='mean')->'Tensor':return MSELoss.apply(x,target,reduction=reduction)
def binary_cross_entropy(x:'Tensor', target:'Tensor', weight=None, reduction='mean')->'Tensor':return BCELoss.apply(x,target,weight=weight,reduction=reduction)
def nll_loss(x:'Tensor', target:'Tensor', weight=None,ignore_index=-100,reduction='mean')->'Tensor':return NllLoss.apply(x,target,weight=weight,ignore_index=ignore_index,reduction=reduction)
def binary_cross_entropy_with_logits(x:'Tensor', target:'Tensor', weight=None,reduction='mean', pos_weight=None)->'Tensor':return BCEWithLogitsLoss.apply(x,target,weight=weight,reduction=reduction,pos_weight=pos_weight)
def cross_entropy(x,target,axis=1,weight=None,ignore_index=-100,reduction='mean'):return nll_loss(log_softmax(x,dim=axis),target,weight=weight,ignore_index=ignore_index,reduction=reduction)



        






