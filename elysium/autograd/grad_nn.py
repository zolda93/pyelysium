from typing import Tuple,Union,List,Optional
from .function import *
from .helper import *
from elysium import np,cp
import elysium as e

class Convolution(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',w:'Tensor',bias:Union['Tensor',None]=None,stride:Optional[Union[Tuple[int,...],int]]=1,padding:Optional[Union[Tuple[int,...],int,str]]=0,dilation:Optional[Union[Tuple[int,...],int]]=1,groups:Optional[int]=1,padding_mode:Optional[str]='zeros')->'Tensor':
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
        if x.requires_grad:x_grad=conv_transpose2d(grad.data,w.data,stride=stride,padding=padding,dilation=dilation,groups=groups,padding_mode=padding_mode,input=x.data,extra_padding=extra_padding)
        if w.requires_grad:w_grad=conv2d_backward_w(x_padded,grad.data, stride, padding, dilation, groups, w.data,padding_mode=padding_mode)[0]
        if bias is not None and bias.requires_grad:b_grad=grad.data.sum((0,2,3))
        x_grad = e.Tensor(x_grad,device=x.device,dtype=x.dtype) if x.requires_grad else None
        w_grad = e.Tensor(w_grad,device=w.device,dtype=w.dtype) if w.requires_grad else None
        b_grad = e.Tensor(b_grad,device=bias.device,dtype=bias.dtype) if bias is not None and bias.requires_grad else None
        return (x_grad,w_grad,b_grad)
class TransposedConvolution(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',w:'Tensor',bias:Union['Tensor',None]=None,stride:Optional[Union[Tuple[int,...],int]]=1,padding:Optional[Union[Tuple[int,...],int]]=0,dilation:Optional[Union[Tuple[int,...],int]]=1,output_padding:Optional[Union[Tuple[int,...],int]]=0,groups:Optional[int]=1,padding_mode:Optional[str]='zeros')->'Tensor':
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
        x,w,bias=ctx.get_saved_tensors()
        stride,padding,dilation,groups=ctx.stride,ctx.padding,ctx.dilation,ctx.groups
        w_grad,grad_padded = conv2d_backward_w(grad.data,x.data ,stride,padding,dilation,groups,w.data,is_transpose=True)
        if x.requires_grad:x_grad=conv2d(grad_padded,w.data,stride=stride,padding=(0,0),dilation=dilation,groups=groups)[0]
        if bias is not None and bias.requires_grad:b_grad=grad.data.sum((0,2,3))
        x_grad=e.Tensor(x_grad,device=x.device,dtype=x.dtype) if x.requires_grad else None
        w_grad = e.Tensor(w_grad,device=w.device,dtype=w.dtype) if w.requires_grad else None
        b_grad = e.Tensor(b_grad,device=bias.device,dtype=bias.dtype) if bias is not None and bias.requires_grad else None
        return (x_grad,w_grad,b_grad)
class MaxPool2DWithIndices(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',kernel_size:Union[int, Tuple[int, int]],stride:Union[int, Tuple[int, int]]=None,padding:Union[int, Tuple[int, int]]=0,dilation:Union[int, Tuple[int, int]]=1,
            return_indices:Optional[bool]=True,ceil_mode:Optional[bool]=False)->'Tensor':
        ctx.save_for_backward(x)
        ctx.kernel_size,ctx.stride,ctx.padding,ctx.dilation,ctx.return_indices,ctx.ceil_mode=kernel_size,stride,padding,dilation,return_indices,ceil_mode
        out,pos=maxpool2d(x.data,kernel_size,stride,dilation,padding,ceil_mode)
        ctx.pos=pos
        return e.Tensor(out,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x=ctx.get_saved_tensors()[0]
        kernel_size,stride,padding,dilation,ceil_mode,pos=ctx.kernel_size,ctx.stride,ctx.padding,ctx.dilation,ctx.ceil_mode,ctx.pos
        x_grad = maxpool2d_backward(x.data,grad.data,pos,kernel_size,stride, padding,dilation,ceil_mode) if x.requires_grad else None
        return (e.Tensor(x_grad,device=x.device,dtype=x.dtype) if x_grad is not None else None,)
class AvgPool2d(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',kernel_size,stride=None,padding=0,ceil_mode=False,count_include_pad=True,divisor_override=None)->'Tensor':
        ctx.save_for_backward(x)
        ctx.kernel_size,ctx.stride,ctx.padding,ctx.ceil_mode=kernel_size,stride,padding,ceil_mode
        out,divisor= avgpool2d_forward(x.data,kernel_size,stride=stride,padding=padding,ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)
        ctx.divisor = divisor
        return e.Tensor(out,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x=ctx.get_saved_tensors()[0]
        kernel_size,stride,padding,ceil_mode,divisor=ctx.kernel_size,ctx.stride,ctx.padding,ctx.ceil_mode,ctx.divisor
        x_grad = avgpool2d_backward(x.data,grad.data,divisor,kernel_size,stride,padding,ceil_mode=ceil_mode) if x.requires_grad else None
        return (e.Tensor(x_grad,device=x.device,dtype=x.dtype) if x_grad is not None else None,)
class ConstantPad2d(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',padding,val)->'Tensor':
        xp = cp if x.device=='gpu' else np
        ctx.save_for_backward(x)
        let,right,top,bottom = convert_padding(padding)
        ctx.padding=padding
        if x.ndim == 3:
            out = xp.pad(x.data[None], ((0, 0), (0, 0), (top, bottom), (left, right)), mode='constant',constant_values=val)[0]
        else:
            out=xp.pad(x.data, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='constant',constant_values=val)
        return e.Tensor(out,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x = ctx.get_saved_tensors()[0]
        left,right,top,bottom=convert_padding(ctx.padding)
        left, right, top, bottom = self.padding
        h_in = grad.shape[-2] - top - bottom
        w_in = grad.shape[-1] - left - right
        return  (e.Tensor(grad.data[..., top:top + h_in, left:left + w_in],device=x.device,dtype=x.dtype) if x.requires_grad else None,)
class ReflectionPad2d(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',padding)->'Tensor':
        xp = cp if x.device=='gpu' else np
        ctx.save_for_backward(x)
        ctx.padding=padding
        left,right,top,bottom=convert_padding(padding)
        if x.ndim == 3:
            out = xp.pad(x.data[None], ((0, 0), (0, 0), (top, bottom), (left, right)), mode='reflect')[0]
        else:
            out=xp.pad(x.data, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='reflect')
        return e.Tensor(out,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x=ctx.get_saved_tensors()[0]
        xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
        left,right,top,bottom=convert_padding(ctx.padding)
        h_in = grad.shape[-2] - top - bottom
        w_in = grad.shape[-1] - left - right
        grad=grad.data
        if top > 0:grad[...,top+1:2*top+1,:] += xp.flip(grad[..., :top,:], axis=2)
        if bottom>0:grad[...,-2*bottom-1:-bottom-1,: ]+=xp.flip(grad[..., -bottom:,:], axis=2)
        if left>0:grad[...,left+1:2*left+1]+=xp.flip(grad[...,:left], axis=3)
        if right>0:grad[...,-2*right-1:-right-1]+=xp.flip(grad[..., -right:], axis=3)
        return (e.Tensor(grad[...,top:top + h_in, left:left + w_in],device=x.device,dtype=x.dtype) if x.requires_grad else None,)
class CircularPad2d(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',padding)->'Tensor':
        xp = cp if x.device=='gpu' else np
        ctx.save_for_backward(x)
        ctx.padding=padding
        left,right,top,bottom=convert_padding(padding)
        if x.ndim == 3:
            out = xp.pad(x.data[None], ((0, 0), (0, 0), (top, bottom), (left, right)), mode='wrap')[0]
        else:
            out=xp.pad(x.data, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='wrap')
        return e.Tensor(out,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x=ctx.get_saved_tensors()[0]
        xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
        left,right,top,bottom=convert_padding(ctx.padding)
        h_in = grad.shape[-2] - top - bottom
        w_in = grad.shape[-1] - left - right
        grad=grad.data
        if top > 0 and bottom > 0:
            grad[..., top:top+bottom, :] += grad[..., -bottom:, :]
            grad[..., -bottom - top:-bottom, :] += grad[...,:top, :]
        elif top > 0 and bottom <= 0:  # Bottom padding is 0
            grad[..., -top:, :] += grad[..., :top, :]
        elif bottom > 0 and top <= 0:  # Top padding is 0
            grad[..., :bottom, :] += grad[..., -bottom:, :]
        if left > 0 and right > 0:
            grad[..., -right - left:-right] += grad[..., :left]
            grad[..., left:left + right] += grad[..., -right:]
        elif left > 0 and right <= 0:  # Right padding is 0
            grad[..., -left:] += grad[..., :left]
        elif right > 0:  # Left padding is 0
            grad[..., :right] += grad[..., -right:]
        return (e.Tensor(grad[:, :, top:top + h_in, left:left + w_in],device=x.device,dtype=x.dtype) if x.requires_grad else None,)
class ReplicationPad2d(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',padding)->'Tensor':
        xp = cp if x.device=='gpu' else np
        ctx.save_for_backward(x)
        ctx.padding=padding
        left,right,top,bottom=convert_padding(padding)
        if x.ndim == 3:
            out = xp.pad(x.data[None], ((0, 0), (0, 0), (top, bottom), (left, right)), mode='edge')[0]
        else:
            out=xp.pad(x.data, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='edge')
        return e.Tensor(out,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x=ctx.get_saved_tensors()[0]
        xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
        left,right,top,bottom=convert_padding(ctx.padding)
        h_in = grad.shape[-2] - top - bottom
        w_in = grad.shape[-1] - left - right
        grad=grad.data
        if top > 0:grad[...,top,:] += xp.sum(grad[..., :top,:],axis=2)
        if bottom >0:grad[...,-bottom-1,: ]+=xp.sum(grad[..., -bottom:,:],axis=2)
        if left>0:grad[...,left]+=xp.sum(grad[..., :left],axis=3)
        if right>0:grad[...,-right-1]+=xp.sum(grad[..., -right:],axis=3)
        return (e.Tensor(grad[:, :, top:top + h_in, left:left + w_in],device=x.device,dtype=x.dtype) if x.requires_grad else None,)
class Embedding(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',weight:'Tensor',padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)->'Tensor':
        ctx.save_for_backward(x,weight)
        xp = cp if weight.device=='gpu' else np
        x.data = x.data.astype(xp.int32)
        ctx.padding_idx,ctx.max_norm,ctx.norm_type,ctx.scale_grad_by_freq,ctx.sparse=padding_idx,max_norm,norm_type,scale_grad_by_freq,sparse
        embeddings = weight.data[x.data]
        if max_norm is not None:
            norms = xp.linalg.norm(embeddings, ord=norm_type, axis=-1, keepdims=True)
            mask = norms > max_norm
            embeddings = xp.where(mask, embeddings * (max_norm / norms), embeddings)
        return e.Tensor(embeddings,requires_grad=weight.requires_grad,device=weight.device,dtype=weight.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x,weight=ctx.get_saved_tensors()
        xp = cp if weight.device=='gpu' else np
        grad_w = xp.zeros_like(weight.data)
        if cp is not None and xp is cp:
            import cupyx
            add_at = cupyx.scatter_add
        else:
            add_at = np.add.at
        if ctx.sparse:
            add_at(grad_w,x.data,grad.data)
            if ctx.padding_idx is not None:grad_w[padding_idx]=0
            return (None,e.Tensor(grad_w,device=weight.device,dtype=weight.dtype) if weight.requires_grad else None)
        else:
            grad = grad.data
            if ctx.scale_grad_by_freq:
                freqs = xp.bincount(x.data, minlength=weight.shape[0]).astype(xp.float32)
                freqs = freqs[x.data][:, None]
                grad = grad / freqs
            add_at(grad_w,x.data,grad)
            if ctx.padding_idx is not None:grad_w[padding_idx]=0
            return (None,e.Tensor(grad_w,device=weight.device,dtype=weight.dtype) if weight.requires_grad else None)
class ReLU(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',inplace:bool=False)->'Tensor':
        ctx.save_for_backward(x)
        if inplace:
            x.data = x.data>0
            return e.Tensor(x.data,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
        return e.Tensor(x.data>0,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x=ctx.get_saved_tensors()[0]
        return (e.Tensor(grad.data * (x.data > 0),device=x.device,dtype=x.dtype) if a.requires_grad else None,)

class Sigmoid(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor')->'Tensor':
        xp = cp if x.device == 'gpu' else np
        out = xp.divide(1,xp.add(1,xp.exp(-x.data),dtype=x.dtype),dtype=x.dtype)
        ctx.out,ctx.device= out,x.device
        return e.Tensor(out,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        grad_x = grad.data * ctx.out*(1-ctx.out)
        return (e.Tensor(grad_x,device=ctx.device,dtype=grad_x.dtype),)
class LogSigmoid(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor')->'Tensor':
        ctx.save_for_backward(x)
        xp = cp if x.device=='gpu' else np
        out = x.data * (x.data < 0) -xp.log1p(xp.exp(-xp.abs(x.data)))
        ctx.out=out
        return e.Tensor(out,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x=ctx.get_saved_tensors()[0]
        xp = cp if x.device=='gpu' else np
        grad_x = grad.data * xp.exp(-x.data + ctx.out)
        return (e.Tensor(grad_x,device=x.device,dtype=x.dtype),)







