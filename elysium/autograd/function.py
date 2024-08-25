from typing import List,Tuple,Union

class Context:
    def __init__(self):
        self.saved_tensors: List['Tensor'] = []
    def save_for_backward(self,*tensors:'Tensor')->None:
        self.saved_tensors.extend(tensors)
    def get_saved_tensors(self)->Tuple['Tensor',...]:
        return tuple(self.saved_tensors)

class Function:
    @staticmethod
    def forward(ctx:Context,*args:'Tensor',**kwargs)->'Tensor':
        raise NotImplementedError
    @staticmethod
    def backward(ctx:Context,*grads:'Tensor')->Tuple[Union['Tensor',None],...]:
        raise NotImplementedError
    @classmethod
    def apply(cls,*args:'Tensor',**kwargs)->'Tensor':
        ctx = Context()
        result = cls.forward(ctx,*args,**kwargs)
        result._ctx = ctx
        ctx.func = cls
        return result

