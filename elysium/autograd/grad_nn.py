from typing import Tuple,Union,List,Optional
from .function import *
from .helper import *
from elysium import np,cp
import elysium as e

class Convolution(Function):
    @staticmethod
    def forward(ctx:Context,
            x:'Tensor',
            w:'Tensor',
            bias:Union['Tensor',None]=None,
            stride:Optional[Union[Tuple[int,...],int]]=1,
            padding:Optioanl[Union[Tuple[int,...],int,str]]0,
            dilation:Optional[Union[Tuple[int,...],int]]=1,
            groups:Optional[int]=1,
            padding_mode:Optional[str]='zeros')->'Tensor':
        if x.__class__ is not w.__class__:
            raise RuntimeError(f'Input type ({x.__class__.__name__}) and weight type ({w.__class__.__name__}) should be the same')
        if bias is not None and x.__class__ is not bias.__class__:
            raise RuntimeError(f'Input type ({x.__class__.__name__}) and bias type ({bias.__class__.__name__}) should be the same')
        if x.ndim != 4:raise RuntimeError(f'Expected 3D (unbatched) or 4D (batched) input to conv2d, 'f'but got input of size: {x.shape}')
        if groups * w.shape[-3] != x.shape[-3]:
            raise RuntimeError(f'Given groups={groups}, weight of size {w.shape}, '
                               f'expected input{x.shape} to have {groups * w.shape[-3]} channels, '
                               f'but got {x.shape[-3]} channels instead')
        ctx.save_for_backward(x,w,bias)
        bias = bias.data if bias is not None else None
        output,x_padded,extra_padding = conv2d(x.data,w.data,bias=bias,stride=stride,padding=padding,dilation=dilation,groups=groups,padding_mode=padding_mode)
        ctx.x_padded,ctx.extra_padding=x_padded,extra_padding
        ctx.stride,ctx.padding,ctx.dilation,ctx.groups,ctx.padding_mode=stride,padding,dilation,groups,padding_mode
        requires_grad=x.requires_grad|w.requires_grad
        return e.Tensor(output,requires_grad=requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x,w,bias=ctx.get_saved_tensors()
        stride,padding,dilation,groups,padding_mode=ctx.stride,ctx.padding,ctx.dilation,ctx.groups,ctx.padding_mode
        x_padded,extra_padding=ctx.x_padded,ctx.extra_padding
        if x.requires_grad:x_grad=conv_transpose2d(grad.data,w.data,stride=stride,padding=padding,dilation=dilation,groups=groups,padding_mode=padding_mode,x=x.data,extra_padding=extra_padding)
        if w.requires_grad:w_grad=conv2d_backward_w(x_padded,grad.data, stride, padding, dilation, groups, w.data,padding_mode=padding_mode)[0]
        if bias is not None and bias.requires_grad:b_grad=grad.data.sum((0,2,3))
        x_grad = e.Tensor(x_grad,device=x.device,dtype=x.dtype) if x.requires_grad else None
        w_grad = e.Tensor(w_grad,device=w.device,dtype=w.dtype) if w.requires_grad else None
        b_grad = e.Tensor(b_grad,device=bias.device,dtype=bias.dtype) if bias is not None and bias.requires_grad else None
        return (x_grad,w_grad,b_grad)
class TransposedConvolution(Function):
    @staticmethod
    def forward(ctx:Context,
            x:'Tensor'
            w:'Tensor'
            bias:Union['Tensor',None]=None,
            stride:Optional[Union[Tuple[int,...],int]]=1,
            padding:Optional[Union[Tuple[int,...],int]]=0,
            dilation:Optional[Union[Tuple[int,...],int]]=1,
            output_padding:Optional[Union[Tuple[int,...],int]]=0,
            groups:Optional[int]=1,
            padding_mode:Optional[str]='zeros')->'Tensor':
        if x.__class__ is not w.__class__:
            raise RuntimeError(f'Input type ({x.__class__.__name__}) and weight type ({w.__class__.__name__}) should be the same')
        if bias is not None and x.__class__ is not bias.__class__:
            raise RuntimeError(f'Input type ({x.__class__.__name__}) and bias type ({bias.__class__.__name__}) should be the same')
        if x.ndim != 4:
            raise RuntimeError(f'Expected 3D (unbatched) or 4D (batched) input to conv_transpose2d, '
                               f'but got input of size: {x.shape}')
        if w.shape[-4] != x.shape[-3]:
            raise RuntimeError(f'Given transposed=1, weight of size {w.shape}, '
                               f'expected input {x.shape} to have {w.shape[-4]} channels, '
                               f'but got {x.shape[-3]} channels instead')
        ctx.save_for_backward(x,w,bias)
        bias = bias.data if bias is not None else None
        if padding_mode != 'zeros':raise ValueError('Only "zeros" padding mode is supported for ConvTranspose2d')
        output= conv_transpose2d(x.data,w.data,bias=bias,stride=stride,padding=padding,dilation=dilation,groups=groups,output_padding=output_padding)
        ctx.stride,ctx.padding,ctx.dilation,ctx.groups=stride,padding,dilation,groups
        requires_grad=x.requires_grad|w.requires_grad
        return e.Tensor(output,requires_grad=requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        pass


        









