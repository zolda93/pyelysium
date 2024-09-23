from typing import Tuple,Union,List,Optional
from .function import *
from .helper import *
from elysium import np,cp
import elysium as e

class Convolution(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',w:'Tensor',bias:Union['Tensor',None]=None,stride:Optional[Union[Tuple[int,...],int]]=1,padding:Optional[Union[Tuple[int,...],int,str]]=0,dilation:Optional[Union[Tuple[int,...],int]]=1,groups:Optional[int]=1,padding_mode:Optional[str]='zeros')->'Tensor':
        if x.data.__class__ is not w.data.__class__:
            raise RuntimeError(f'Input data type ({x.data._class__}) and weight data type ({w.data.__class__}) should be the same')
        if bias is not None and x.data.__class__ is not bias.data.__class__:
            raise RuntimeError(f'Input data type ({x.data.__class__}) and bias data type ({bias.data.__class__}) should be the same')
        if x.ndim != 4:raise RuntimeError(f'Expected 3D (unbatched) or 4D (batched) input to conv2d, 'f'but got input of size: {x.shape}')
        if groups * w.shape[-3] != x.shape[-3]:
            raise RuntimeError(f'Given groups={groups}, weight of size {w.shape}, '
                               f'expected input{x.shape} to have {groups * w.shape[-3]} channels, '
                               f'but got {x.shape[-3]} channels instead')
        if bias is not None:
            ctx.save_for_backward(x,w,bias)
        else:
            ctx.save_for_backward(x,w)
        bias = bias.data if bias is not None else None
        output,x_padded,extra_padding = conv2d(x.data,w.data,bias=bias,stride=stride,padding=padding,dilation=dilation,groups=groups,padding_mode=padding_mode)
        ctx.x_padded,ctx.extra_padding=x_padded,extra_padding
        ctx.stride,ctx.padding,ctx.dilation,ctx.groups,ctx.padding_mode=stride,padding,dilation,groups,padding_mode
        requires_grad=x.requires_grad|w.requires_grad
        return e.Tensor(output,requires_grad=requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        #x,w,bias=ctx.get_saved_tensors()
        ts = ctx.get_saved_tensors()
        if len(ts) == 3:
            x,w,bias = ts
        else:
            x,w,bias = ts[0],ts[1],None
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
        if x.data.__class__ is not w.data.__class__:
            raise RuntimeError(f'Input data type ({x.data.__class__}) and weight data type ({w.data.__class__}) should be the same')
        if bias is not None and x.data.__class__ is not bias.data.__class__:
            raise RuntimeError(f'Input data type ({x.data.__class__}) and bias data type ({bias.data.__class__}) should be the same')
        if x.ndim != 4:
            raise RuntimeError(f'Expected 3D (unbatched) or 4D (batched) input to conv_transpose2d, '
                               f'but got input of size: {x.shape}')
        if w.shape[-4] != x.shape[-3]:
            raise RuntimeError(f'Given transposed=1, weight of size {w.shape}, '
                               f'expected input {x.shape} to have {w.shape[-4]} channels, '
                               f'but got {x.shape[-3]} channels instead')
        #ctx.save_for_backward(x,w,bias)
        if bias is not None:
            ctx.save_for_backward(x,w,bias)
        else:
            ctx.save_for_backward(x,w)
        bias = bias.data if bias is not None else None
        if padding_mode != 'zeros':raise ValueError('Only "zeros" padding mode is supported for ConvTranspose2d')
        output= conv_transpose2d(x.data,w.data,bias=bias,stride=stride,padding=padding,dilation=dilation,groups=groups,output_padding=output_padding)
        ctx.stride,ctx.padding,ctx.dilation,ctx.groups=stride,padding,dilation,groups
        requires_grad=x.requires_grad|w.requires_grad
        return e.Tensor(output,requires_grad=requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        #x,w,bias=ctx.get_saved_tensors()
        ts = ctx.get_saved_tensors()
        if len(ts) == 3:
            x,w,bias = ts
        else:
            x,w,bias = ts[0],ts[1],None
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
        left,right,top,bottom = convert_padding(padding)
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
        xp = cp if x.device == 'gpu' else np
        if inplace:
            x.data[...] = xp.maximum(x.data,0)
            return e.Tensor(x.data,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
        return e.Tensor(xp.maximum(x.data,0),requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x=ctx.get_saved_tensors()[0]
        return (e.Tensor(grad.data * (x.data>0),device=x.device,dtype=x.dtype) if x.requires_grad else None,)

class Sigmoid(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor')->'Tensor':
        ctx.save_for_backward(x)
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
class Tanh(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor')->'Tensor':
        ctx.save_for_backward(x)
        xp = cp if x.device=='gpu' else np
        out = xp.tanh(x.data)
        ctx.out = out
        return e.Tensor(out,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x = ctx.get_saved_tensors()[0]
        return (e.Tensor(grad.data * (1 - ctx.out * ctx.out),device=x.device,dtype=x.dtype),)
class LeakyReLU(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',negative_slope=0.01, inplace=False)->'Tensor':
        ctx.save_for_backward(x)
        ctx.negative_slope=negative_slope
        xp = cp if x.device=='gpu' else np
        if inplace:
            x.data[:] = xp.maximum(0,x.data) + negative_slope * xp.minimum(0, x.data)
            ctx.out=x.data
            return e.Tensor(x.data,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
        out = xp.maximum(0, x.data) + negative_slope * xp.minimum(0, x.data)
        ctx.out = out
        return e.Tensor(out,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x = ctx.get_saved_tensors()[0]
        xp = cp if x.device=='gpu' else np
        grad_x = xp.where(ctx.out > 0,1,ctx.negative_slope) * grad.data
        return (e.Tensor(grad_x,device=x.device,dtype=x.dtype),)
class Softmax(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',dim=None)->'Tensor':
        ctx.save_for_backward(x)
        xp = cp if x.device=='gpu' else np
        aug = x.data - xp.max(x.data,axis=dim,keepdims=True)
        exp = xp.exp(aug)
        sum_exp = xp.sum(exp,axis=dim,keepdims=True)
        ctx.out, ctx.dim= exp/sum_exp,dim
        return e.Tensor(exp/sum_exp,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x=ctx.get_saved_tensors()[0]
        xp = cp if x.device=='gpu' else np
        grad_x = (grad.data - xp.sum((grad.data * ctx.out),axis=ctx.dim,keepdims=True)) * ctx.out
        return (e.Tensor(grad_x,device=x.device,dtype=x.dtype),)
class LogSoftmax(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',dim=None)->'Tensor':
        ctx.save_for_backward(x)
        xp = cp if x.device=='gpu' else np
        aug = x.data - xp.max(x.data,axis=dim,keepdims=True)
        exp = xp.exp(aug)
        sum_exp = xp.sum(exp,axis=dim,keepdims=True)
        log_sum_exp = xp.log(sum_exp)
        ctx.dim,ctx.softmax = dim,exp/sum_exp
        return e.Tensor(aug - log_sum_exp,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x = ctx.get_saved_tensors()[0]
        xp = cp if x.device=='gpu' else np
        grad_x = grad.data - xp.sum(grad.data,axis=ctx.dim,keepdims=True) * ctx.softmax
        return (e.Tensor(grad_x,device=x.device,dtype=x.dtype),)
class L1Loss(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',target:'Tensor',reduction:Optional[str]='mean')->'Tensor':
        assert x.shape == target.shape ,f"target shape {target.shape} must match input shape {x.shape}"
        ctx.save_for_backward(x,target)
        ctx.reduction=reduction
        xp = cp if x.device=='gpu' else np
        loss = xp.abs(x.data - target.data)
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:
            raise ValueError("{} is not a valid value for reduction".format(reduction))
        return e.Tensor(loss,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x,target=ctx.get_saved_tensors()
        xp = cp if x.device=='gpu' else np
        grad_x = xp.sign(x.data - target.data)
        if ctx.reduction == 'mean':grad_x = xp.divide(grad_x,x.data.size)
        return (e.Tensor(grad_x,device=x.device,dtype=x.dtype),None)
class MSELoss(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',target:'Tensor',reduction:Optional[str]='mean')->'Tensor':
        assert x.shape == target.shape ,f"target shape {target.shape} must match input shape {x.shape}"
        ctx.save_for_backward(x,target)
        ctx.reduction=reduction
        loss = (x.data - target.data)**2
        if reduction=='mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:
            raise ValueError("{} is not a valid value for reduction".format(reduction))
        return e.Tensor(loss,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x,target = ctx.get_saved_tensors()
        grad_x = 2 * grad.data * (x.data - target.data)
        xp = cp if x.device=='gpu' else np
        if ctx.reduction == 'mean':grad_x = xp.divide(grad_x,x.data.size)
        return (e.Tensor(grad_x,device=x.device,dtype=x.dtype),None)
class BCELoss(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',target:'Tensor',weight=None,reduction='mean')->'Tensor':
        assert x.shape == target.shape ,f"target shape {target.shape} must match input shape {x.shape}"
        ctx.save_for_backward(x,target)
        xp = cp if x.device=='gpu' else np
        # Clamp input to prevent log(0) issues and to ensure numerical stability
        eps = 1e-12
        x_clamped = xp.clip(x.data, eps, 1.0 - eps)
        # Compute the binary cross entropy loss
        loss = - (target.data * xp.log(x_clamped) + (1 - target.data) * xp.log(1 - x_clamped))
        if weight is not None:loss*=weight.data
        if reduction=='mean':
            loss=loss.mean()
        elif reduction=='sum':
            loss=loss.sum()
        else:
            raise ValueError("{} is not a valid value for reduction".format(reduction))
        ctx.reduction,ctx.x_clamped,ctx.weight = reduction,x_clamped,weight
        return e.Tensor(loss,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x,target = ctx.get_saved_tensors()
        xp = cp if x.device=='gpu' else np
        grad_x = - (target.data / ctx.x_clamped) + (1 - target.data) / (1 - ctx.x_clamped)
        if ctx.weight is not None:grad_x*=ctx.weight.data
        if ctx.reduction == 'mean':grad_x /= x.data.size
        return (e.Tensor(grad_x,device=x.device,dtype=x.dtype),None)
class NllLoss(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',target:'Tensor',weight=None,reduction='mean',ignore_index=-100)->'Tensor':
        xp = cp if x.device=='gpu' else np
        target.data = target.data.astype(xp.int32)
        ctx.save_for_backward(x,target)
        # Input should be log-probabilities (logits after LogSoftmax)
        batch_size = target.shape[0]
        nll_loss = -x.data[xp.arange(batch_size),target.data]# Gather the log-probabilities corresponding to the target classes
        if weight is not None:
            class_weights = weight.data[target.data]
            nll_loss*=class_weights
        if ignore_index is not None:
            mask = (target.data != ignore_index)
            nll_loss = nll_loss[mask]
        if reduction   == 'mean':
            nll_loss = nll_loss.mean()
        elif reduction == 'sum':
            nll_loss = nll_loss.sum()
        else:
            raise ValueError("{} is not a valid value for reduction".format(reduction))
        ctx.weight,ctx.reduction,ctx.ignore_index = weight,reduction,ignore_index
        return e.Tensor(nll_loss,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x,target=ctx.get_saved_tensors()
        xp = cp if x.device=='gpu' else np
        batch_size = target.shape[0]
        # Create a mask for the valid targets (not ignored)
        valid_mask = -(target.data != ctx.ignore_index).astype(xp.float32)
        grad_x = xp.zeros_like(x.data)
        # Use the valid_mask to update the gradient only for valid targets
        grad_x[xp.arange(batch_size), target.data] =  valid_mask
        # Apply class weights if provided
        if ctx.weight is not None:grad_x *= ctx.weight.data[xp.newaxis, :]
        if ctx.reduction == 'mean':grad_x /= batch_size
        return (e.Tensor(grad_x,device=x.device,dtype=x.dtype),None)
class BCEWithLogitsLoss(Function):
    @staticmethod
    def forward(ctx:Context,x:'Tensor',target:'Tensor',weight=None,reduction='mean',pos_weight=None)->'Tensor':
        ctx.save_for_backward(x,target)
        xp = cp if x.device=='gpu' else np
        # Stable log-sum-exp trick for sigmoid
        log_sigmoid = -xp.logaddexp(0, -x.data)
        log_one_minus_sigmoid = -x.data - xp.logaddexp(0, -x.data)
        if pos_weight is not None:
            loss = -(pos_weight.data * target.data * log_sigmoid + (1 - target.data) * log_one_minus_sigmoid)
        else:
            loss = -(target.data * log_sigmoid + (1 - target.data) * log_one_minus_sigmoid)
        if weight is not None:loss *= weight.data
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:
            raise ValueError("{} is not a valid value for reduction".format(reduction))
        ctx.weight,ctx.reduction,ctx.pos_weight=weight,reduction,pos_weight
        return e.Tensor(loss,requires_grad=x.requires_grad,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        x,target = ctx.get_saved_tensors()
        xp = cp if x.device=='gpu' else np
        # Sigmoid function without overflow/underflow
        sigmoid_x = 1 / (1 + xp.exp(-x.data))
        if ctx.pos_weight is not None:
            grad_x = ctx.pos_weight.data * target.data * (sigmoid_x - 1) + (1 - target.data) * sigmoid_x
        else:
            grad_x = target.data * (sigmoid_x - 1) + (1 - target.data) * sigmoid_x
        if ctx.weight is not None:grad_x *= ctx.weight.data
        if ctx.reduction == 'mean':grad_x /= x.data.size
        return (e.Tensor(grad_x,device=x.device,dtype=x.dtype),None)
    
