from typing import Tuple,Union,List,Optional
from .base import *
from elysium import np,cp
import elysium as e


def sum_to_shape(x,shape:Tuple[int,...])->'Tensor':
    dims_to_sum = tuple(i for i,(s1,s2) in enumerate(zip(x.shape,shape)) if s1!=s2)
    return e.Tensor(x.sum(axis=dims_to_sum,keepdims=(len(dims_to_sum)>0)),dtype=x.dtype)
class Neg(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor')->'Tensor':
        ctx.save_for_backward(a)
        return e.Tensor(-a.data,requires_grad=a.requires_grad,device=a.device)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None]]:
        a = ctx.saved_tensors[0]
        return (-grad,) if a.requires_grad else (None,)


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
        grad_a = e.Tensor(grad.data@xp.swapaxes(b.data,-1,-2),device=a.device,dtype=a.dtype) if a.requires_grad else None
        grad_b = e.Tensor(xp.swapaxes(a.data,-1,-2)@grad.data,device=b.device,dtype=b.dtype) if b.requires_grad else None
        return (grad_a,grad_b)

class Sum(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',axis:Union[Tuple[int,...],None]=None,keepdim:Optional[bool]=False)->'Tensor':
        if axis is None:axis = tuple(range(a.ndim))
        axis = (axis,) if isinstance(axis,int) else axis
        ctx.save_for_backward(a)
        ctx.axis,ctx.keepdim = axis,keepdim
        xp = cp if a.device == 'gpu' else np
        return e.Tensor(xp.sum(a.data,axis=axis,keepdims=keepdim),requires_grad=a.requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Union['Tensor',None]:
        a = ctx.get_saved_tensors()[0]
        axis = ctx.axis
        keepdim = ctx.keepdim
        xp = cp if a.device == 'gpu' else np
        grad = grad.data
        if not keepdim:grad = xp.expand_dims(grad,axis=axis)
        strides = list(grad.strides)
        for i in axis:strides[i] = 0
        return (e.Tensor(xp.lib.stride_tricks.as_strided(grad,shape=a.shape,strides=strides),device=a.device,dtype=a.dtype),) if a.requires_grad else None





