from typing import Tuple,Union,List,Optional
from .base import *
from elysium import np,cp
import elysium as e

def sum_to_shape(x,shape:Tuple[int,...])->'Tensor':
    xp = cp if x.device == 'gpu' else np
    dims_to_sum = tuple(i for i,(s1,s2) in enumerate(zip(x.shape,shape)) if s1!=s2)
    return e.Tensor(xp.sum(x,axis=dim_to_sum,keepdims=(len(dims_to_sum)>0)),device=x.device,dtype=x.dtyipe)
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
    def forward(ctx:Context,a:'Tensor',b:'Tensor')->'Tensor':
        assert a.device == b.device,f'tensors must be in the same device got {a.device},{b.device}'
        ctx.save_for_backward(a,b)
        requires_grad = a.requires_grad|b.requires_grad
        return e.Tensor(a.data + b.data,requires_grad=requires_grad,device=a.device)
    @staticmethod
    def bacwkard(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a,b = ctx.get_saved_tensors()
        grad_a = sum_to_shape(grad,a.shape) if a.requires_grad else None
        grad_b = sum_to_shape(grad,b.shape) if b.requires_grad else None
        return (grad_a,grad_b)

class Mul(Function):
    @staticmethod
    def forward(ctx:Context,a:'Tensor',b:'Tensor')->'Tensor':
        assert a.device == b.device,f'tensors must be in the same device got {a.device},{b.device}'
        ctx.save_for_backward(a,b)
        requires_grad = a.requires_grad|b.requires_grad
        return e.Tensor(a.data*b.data,requires_grad=requires_grad,device=a.device,dtype=a.dtype)
    @staticmethod
    def backward(ctx:Context,grad:'Tensor')->Tuple[Union['Tensor',None],...]:
        a,b = ctx.get_saved_tensors()
        grad_a = sum_to_shape(grad.data * b.data ,a.shape) if a.requires_grad else None
        grad_b = sum_to_shape(grad.data * a.data ,b.shape) if b.requires_grad else None
        return (grad_a,grad_b)

