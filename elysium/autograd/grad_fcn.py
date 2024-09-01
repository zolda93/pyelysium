from typing import Tuple,Union,List,Optional
from .function import *
from elysium import np,cp
import elysium as e


def sum_to_shape(x,shape:Tuple[int,...])->'Tensor':
    device = 'gpu' if x.__class__ is cp.ndarray else 'cpu'
    dims_to_sum = tuple(i for i, (s1, s2) in enumerate(zip(x.shape, shape)) if s1 != s2)
    return e.Tensor(x.sum(axis=tuple(range(x.ndim - len(shape))),keepdims=(len(dims_to_sum)>0)),device=device,dtype=x.dtype)
class Neg(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor')->'Tensor':
        ctx.save_for_backward(a)
        return e.Tensor(-a.data,requires_grad=a.requires_grad,device=a.device)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.saved_tensors[0]
        return (-grad if a.requires_grad else None,)
class Add(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',b:'Tensor',inplace:Optional[bool]=False)->'Tensor':
        ctx.save_for_backward(a,b)
        requires_grad = a.requires_grad|b.requires_grad
        if inplace:
            a.data+=b.data
            return e.Tensor(a.data,requires_grad,device=a.device,dtype=a.dtype)
        return e.Tensor(a.data + b.data,requires_grad=requires_grad,device=a.device)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a,b = ctx.get_saved_tensors()
        grad_a = sum_to_shape(grad.data,a.shape) if a.requires_grad else None
        grad_b = sum_to_shape(grad.data,b.shape) if b.requires_grad else None
        return (grad_a,grad_b)
class Sub(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',b:'Tensor',inplace:Optional[bool]=False)->'Tensor':
        ctx.save_for_backward(a,b)
        requires_grad = a.requires_grad|b.requires_grad
        if inplace:
            a.data-=b.data
            return e.Tensor(a.data,requires_grad,device=a.device,dtype=a.dtype)
        return e.Tensor(a.data - b.data,requires_grad=requires_grad,device=a.device)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a,b = ctx.get_saved_tensors()
        grad_a = sum_to_shape(grad.data,a.shape) if a.requires_grad else None
        grad_b = sum_to_shape(-grad.data,b.shape) if b.requires_grad else None
        return (grad_a,grad_b)
class Mul(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',b:'Tensor',inplace:Optional[bool]=False)->'Tensor':
        ctx.save_for_backward(a,b)
        requires_grad = a.requires_grad|b.requires_grad
        if inplace:
            a.data*=b.data
            return e.Tensor(a.data,requires_grad=require_grad,device=a.device,dtype=a.dtype)
        return e.Tensor(a.data*b.data,requires_grad=requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a,b = ctx.get_saved_tensors()
        grad_a = sum_to_shape(grad.data * b.data ,a.shape) if a.requires_grad else None
        grad_b = sum_to_shape(grad.data * a.data ,b.shape) if b.requires_grad else None
        return (grad_a,grad_b)
class Div(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',b:'Tensor',inplace:Optional[bool]=False)->'Tensor':
        ctx.save_for_backward(a,b)
        requires_grad = a.requires_grad|b.requires_grad
        if inplace:
            a.data/=(b.data + 1e-8)
            return e.Tensor(a.data,requires_grad=requires_grad,device=a.device,dtype=a.dtype)
        return e.Tensor(a.data / (b.data + 1e-8),requires_grad=requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a,b = ctx.get_saved_tensors()
        grad_a = sum_to_shape(grad.data / b.data,a.shape) if a.requires_grad else None
        grad_b = sum_to_shape(-(grad.data * a.data) / (b.data + 1e-8)**2,b.shape) if b.requires_grad else None
        return (grad_a,grad_b)
class Pow(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',b:'Tensor',inplace:Optional[bool]=False)->'Tensor':
        ctx.save_for_backward(a,b)
        if inplace:
            a.data**=b.data
            return Tensor(a.data,requires_grad=a.requires_grad|b.requires_grad,device=a.device,dtype=a.dtype)
        return e.Tensor(a.data**b.data,requires_grad=a.requires_grad|b.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a,b= ctx.get_saved_tensors()
        xp = cp if a.device == 'gpu' else np
        grad_a = sum_to_shape(grad.data * b.data * (a.data ** (b.data - 1)),a.shape) if a.requires_grad else None
        grad_b = sum_to_shape(grad.data * (a.data ** b.data) * xp.log(a.data),b.shape) if b.requires_grad else None
        return (grad_a, grad_b)
class Sqrt(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',inplace:Optional[bool]=False)->'Tensor':
        ctx.save_for_backward(a)
        if inplace:
            a.data**=(1/2)
            return e.Tensor(a.data,requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
        return e.Tensor(a.data**(1/2),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        xp = cp if a.device == 'gpu' else np
        grad_data = xp.ascontiguousarray(grad.data)
        out = 0.5 * grad_data / xp.sqrt(a.data)
        return (e.Tensor(out,device=a.device,dtype=a.dtype) if a.requires_grad else None,) 
class Exp(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor')->'Tensor':
        ctx.save_for_backward(a)
        xp = cp if a.device == 'gpu' else np
        return e.Tensor(xp.exp(a.data),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        xp = cp if a.device == 'gpu' else np
        return (e.Tensor(grad.data * xp.exp(a.data),device=a.device,dtype=a.dtype) if a.requires_grad else None,)
class Log(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor')->'Tensor':
        ctx.save_for_backward(a)
        xp = cp if a.device == 'gpu' else np
        return e.Tensor(xp.log(a.data),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        return (e.Tensor(grad.data / a.data,device=a.device,dtype=a.dtype) if a.requires_grad else None,)
class Cos(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor')->'Tensor':
        ctx.save_for_backward(a)
        xp = cp if a.device == 'gpu' else np
        return e.Tensor(xp.cos(a.data),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        xp = cp if a.device == 'gpu' else np
        return (e.Tensor(-grad.data * xp.sin(a.data),device=a.device,dtype=a.dtype) if a.requires_grad else None,)
class Sin(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor')->'Tensor':
        ctx.save_for_backward(a)
        xp = cp if a.device == 'gpu' else np
        return e.Tensor(xp.sin(a.data),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        xp = cp if a.device == 'gpu' else np
        return (e.Tensor(grad.data * xp.cos(a.data),device=a.device,dtype=a.dtype) if a.requires_grad else None,)
class Tan(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor')->'Tensor':
        ctx.save_for_backward(a)
        xp = cp if a.device == 'gpu' else np
        return e.Tensor(xp.tan(a.data),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        xp = cp if a.device == 'gpu' else np
        return (e.Tensor(grad.data/(xp.cos(a.data)**2),device=a.device,dtype=a.dtype) if a.requires_grad else None,)
class Mv(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',b:'Tensor')->'Tensor':
        if a.ndim!=2:raise RuntimeError('first tensor must be a matrix')
        if b.ndim!=1:raise RuntimeError('second tenssor must be a vector')
        ctx.save_for_backward(a,b)
        return e.Tensor(a.data@b.data,requires_grad=a.requires_grad|b.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a,b = ctx.get_saved_tensors()
        grad_a = e.Tensor(grad.data[:,None]@b.data[None],device=a.device,dtype=a.dtype) if a.requires_grad else None
        grad_b = e.Tensor(a.data.T@grad.data,device=b.device,dtype=b.dtype) if b.requires_grad else None
        return (grad_a,grad_b)
class Mm(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',b:'Tensor')->'Tensor':
        if a.ndim!=2:raise RuntimeError('first argument must be a matrix!')
        if b.ndim!=2:raise RuntimeError('second argument must be a matrix!')
        ctx.save_for_backward(a,b)
        return e.Tensor(a.data@b.data,requires_grad=a.requires_grad|b.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a,b = ctx.get_saved_tensors()
        grad_a = e.Tensor(grad.data@b.data.T,device=a.device,dtype=a.dtype) if a.requires_grad else None
        grad_b = e.Tensor(a.data.T@grad.data,device=b.device,dtype=b.dtype) if b.requires_grad else None
        return (grad_a,grad_b)
class Bmm(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',b:'Tensor')->'Tensor':
        ctx.save_for_backward(a,b)
        return e.Tensor(a.data@b.data,requires_grad=a.requires_grad|b.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a,b = ctx.get_saved_tensors()
        xp = cp if a.device == 'gpu' else np
        grad_a = sum_to_shape(grad.data@xp.swapaxes(b.data,-1,-2),a.shape) if a.requires_grad else None
        grad_b = sum_to_shape(xp.swapaxes(a.data,-1,-2)@grad.data,b.shape) if b.requires_grad else None
        return (grad_a,grad_b)
class Sum(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',axis:Union[Tuple[int,...],None]=None,keepdim:Optional[bool]=False)->'Tensor':
        if axis is None:axis = tuple(range(a.ndim))
        axis = (axis,) if isinstance(axis,int) else axis
        ctx.save_for_backward(a)
        ctx.axis,ctx.keepdim = axis,keepdim
        return e.Tensor(a.data.sum(axis=axis,keepdims=keepdim),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        axis = ctx.axis
        keepdim = ctx.keepdim
        xp = cp if a.device == 'gpu' else np
        grad = grad.data
        if not keepdim:grad = xp.expand_dims(grad,axis=axis)
        strides = list(grad.strides)
        for i in axis:strides[i] = 0
        return (e.Tensor(xp.lib.stride_tricks.as_strided(grad,shape=a.shape,strides=strides),device=a.device,dtype=a.dtype) if a.requires_grad else None,) 
class Mean(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',axis:Union[Tuple[int,...],None]=None,keepdims:Optional[bool]=False)->'Tensor':
        if axis is None:axis=tuple(range(a.ndim))
        axis = (axis,) if isinstance(axis,int) else axis
        ctx.axis,ctx.keepdims = axis,keepdims
        ctx.save_for_backward(a)
        return e.Tensor(a.data.mean(axis=axis,keepdims=keepdims),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        axis,keepdims = ctx.axis,ctx.keepdims
        grad = grad.data
        xp = cp if a.device == 'gpu' else np
        divisor = xp.prod(xp.array([a.shape[i] for i in axis])) if axis is not None else a.size
        if not keepdims:grad = xp.expand_dims(grad,axis=axis)
        new_strides = tuple(0 if i in axis else s for i, s in enumerate(grad.strides))
        out = xp.lib.stride_tricks.as_strided(xp.divide(grad, divisor,dtype=grad.dtype), shape=a.shape, strides=new_strides)
        return (e.Tensor(out,device=a.device,dtype=a.dtype) if a.requires_grad else None,)
class View(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',shape)->'Tensor':
        ctx.save_for_backward(a)
        return e.Tensor(a.data.reshape(shape),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        return (e.Tensor(grad.data.reshape(a.shape),device=a.device,dtype=a.dtype) if a.requires_grad else None,) 
class Squeeze(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',dim:Union[Tuple[int,...],None]=None)->'Tensor':
        ctx.save_for_backward(a)
        if dim is None:dim = tuple(range(a.ndim))
        dim = (dim,) if isinstance(dim,int) else dim
        squeeze_dims = tuple(i for i in dim if a.shape[i] == 1)
        ctx.dim = squeeze_dims
        return e.Tensor(a.data.squeeze(axis=squeeze_dims),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        dim = ctx.dim
        xp = cp if a.device=='gpu' else np
        return (e.Tensor(xp.expand_dims(grad.data,axis=dim),device=a.device,dtype=a.dtype) if a.requires_grad else None,) 
class Unsqueeze(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',dim:Tuple[int,...])->'Tensor':
        ctx.save_for_backward(a)
        ctx.dim = dim
        xp = cp if a.device=='gpu' else np
        return e.Tensor(xp.expand_dims(a.data,dim),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        dim = ctx.dim
        return (e.Tensor(grad.data.squeeze(axis=dim),device=a.device,dtype=a.dtype) if a.requires_grad else None,)
class Index(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',key)->'Tensor':
        ctx.save_for_backward(a)
        ctx.key=key
        return e.Tensor(a.data[key],requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a= ctx.get_saved_tensors()[0]
        key=ctx.key
        xp = cp if a.device=='gpu' else np
        grad_a = xp.zeros_like(a.data)
        if xp is cp:
            import cupyx
            add_at = cupyx.scatter_add
        else:
            add_at = np.add.at
        add_at(grad_a,key,grad.data)
        return (e.Tensor(grad_a,device=a.device,dtype=a.dtype) if a.requires_grad else None,)
class Transpose(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',dim0:int,dim1:int)->'Tensor':
        ctx.save_for_backward(a)
        ctx.dim0,ctx.dim1 = dim0,dim1
        xp = cp if a.device=='gpu' else np
        return e.Tensor(xp.swapaxes(a.data,dim0,dim1),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        dim0,dim1=ctx.dim0,ctx.dim1
        xp = cp if a.device=='gpu' else np
        return (e.Tensor(xp.swapaxes(grad.data,dim0,dim1),device=a.device,dtype=a.dtype)if a.requires_grad else None,)
class Permute(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',dim1:Tuple[int,...])->'Tensor':
        ctx.save_for_backward(a)
        ctx.dims=dims
        xp = cp if a.device=='gpu' else np
        return e.Tensor(xp.transpose(a.data,dims),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a=ctx.get_saved_tensors()[0]
        dims = ctx.dims
        xp = cp if a.device=='gpu' else np
        return (e.Tensor(xp.transpose(grad.data,axes=np.argsort(dims)),device=a.device,dtype=grad.dtype) if a.requires_grad else None,)
class Expand(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',shape)->'Tensor':
        ctx.save_for_backward(a)
        shape = tuple([s if shape[dim] == -1 else shape[dim]  for dim,s in enumerate(a.shape)])
        xp = cp if a.device == 'gpu' else np
        return e.Tensor(xp.broadcast_to(a.data,shape),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        num_new_dims = len(grad.shape) - len(a.shape)
        input_shape = (1,) * num_new_dims + a.shape
        grad_a= grad.data.sum(axis=tuple(i for i, (s, d) in enumerate(zip(a.shape, grad.shape)) if s == 1)).reshape(a.shape[-len(a.shape):])
        return (e.Tensor(grad_a,device=a.device) if a.requires_grad else None,)
class Repeat_Interleave(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',repeats:Union[int,List[int]],axis:Union[int,None]=None)->'Tensor':
        ctx.save_for_backward(a)
        xp=cp if a.device=='gpu' else np
        ctx.repeats,ctx.axis=repeats,axis
        if isinstance(repeats,list):repeats=xp.array(repeats)
        return e.Tensor(xp.repeat(a.data,repeats,axis=axis),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        repeats,axis=ctx.repeats,ctx.axis
        xp=cp if a.device=='gpu' else np
        if axis is None:
            if isinstance(repeats,list):
                grad_a,idx=[],0
                for i,rep in enumerate(repeats):
                    grad_a.append(xp.sum(grad.data[idx:idx+rep]))
                    idx+=rep
                grad_a = xp.array(grad_a).reshape(a.shape)
                return (e.Tensor(grad_a,device=a.device,dtype=a.dtype) if a.reuqires_grad else None,)
            else:
                grad_a = xp.add.reduceat(grad.data,xp.arange(0,grad.data.size,repeats)).reshape(a.shape)
                return (e.Tensor(grad_a,device=a.device,dtype=a.dtype) if a.requires_grad else None,)
        else:
            if isinstance(repeats,int):repeats=[repeats]*a.shape[axis]
            grad_a = xp.concatenate([
                xp.sum(t,axis=axis,keepdims=True)
                for t in xp.split(grad.data,indices_or_sections=xp.cumsum(repeats)[:-1],axis=axis)
                ],axis=axis)
            return (e.Tensor(grad_a,device=a.device,dtype=a.dtype) if a.requires_grad else None,)
class Repeat(Function):
    @staticmethod
    def forward(ctx,a:'Tensor',reps)->'Tensor':
        ctx.save_for_backward(a)
        ctx.reps = reps
        xp = cp if a.device=='gpu' else np
        return e.Tensor(xp.tile(a.data,reps),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def bacwkard(ctx,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a = ctx.get_saved_tensors()[0]
        reps = ctx.reps
        base_shape = (1,) *(len(reps) - a.ndim) + a.shape
        new_shape = [x for b in base_shape for x in [1, b]]
        expand_shape = [x for rs in zip(reps, base_shape) for x in rs]
        grad = grad.data.reshape(expand_shape)
        grad_a=grad.sum(axis=tuple(i for i, (s, d) in enumerate(zip(new_shape, grad.shape)) if s == 1)).reshape(a.shape)
        return (e.Tensor(grad_a,device=a.device,dtype=a.dtype) if a.requires_grad else None,)

