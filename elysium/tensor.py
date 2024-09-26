from typing import Tuple,List,Union,Optional,TypeVar
from elysium import np,cp,ConstType,Dtype,TensorType,no_grad
from .autograd.grad_fcn import *
T = TypeVar('T',bound='Tensor')

class Tensor:
    __slots__ = ('data', '_requires_grad', 'grad','_ctx', 'device')
    __deletable__ = ('_ctx',)
    def __init__(self,
            data:TensorType,
            requires_grad:Optional[bool]=False,
            device:Optional[Union[str,tuple,list]]='cpu',
            dtype:Optional[Dtype] = np.float32)->None:
        self._requires_grad = requires_grad and not no_grad._enabled
        self.grad = None
        self._ctx = None
        self.device = device
        
        if device == 'gpu' and cp is not None:
            self.data = self._initialize_data(data,cp,dtype)
        elif device == 'cpu':
            self.data = self._initialize_data(data,np,dtype)
        else:
            raise TypeError(f"Unsupported device type: {device}")

    def _initialize_data(self,data:Optional[TensorType],lib,dtype:Dtype):
        if data is None:
            return lib.array([],dtype=dtype)
        elif isinstance(data,(float,int,bool,list)):
            return lib.array(data,dtype=dtype)
        elif lib is cp:
            if isinstance(data,np.ndarray):return lib.array(data,dtype=dtype)
            return data.astype(dtype)            
        elif lib is np:
            if cp is not None and isinstance(data,cp.ndarray):return data.get()
            return data.astype(dtype)
        else:
            raise TypeError(f"Unsupported data type for Tensor initialization: {type(data)}")

    #def __del__(self):
        #try:
           # if self._ctx is not None:
         #       self._ctx = None
        #except Exception as e:
            #print(f"Exception in __del__ of Tensor: {e}")

    @property
    def requires_grad(self)->bool:return self._requires_grad
    @requires_grad.setter
    def requires_grad(self,value:bool)->None:
        if not self._is_floating_point():
            raise ValueError("Cannot require gradients for non-floating-point tensor types.")
        self._requires_grad = value
    def _is_floating_point(self)->bool:
        xp = cp if self.device == 'gpu' else np
        return self.data.dtype in [xp.float16,xp.float32,xp.float64]
    @property
    def shape(self)->Tuple[int,...]:return self.data.shape
    @property
    def dtype(self)->Dtype:return self.data.dtype
    @property
    def strides(self):return self.data.strides
    @property
    def ndim(self)->int:return self.data.ndim
    def astype(self,dtype):
        self.data = self.data.astype(dtype)
        return self
    def item(self):return self.data.item()
    def numel(self):return self.data.size
    def to(self,device:str)->'Tensor':
        if device == self.device:
            return self
        elif device == 'cpu':
            return self.cpu()
        elif device == 'gpu' and cp is not None:
            return self.cuda()
        else:
            raise ValueError(f"Unsupported device type: {device}")
    def cuda(self)->'Tensor':
        if self.device == 'gpu':return self
        if cp is None:raise RuntimeError('Cupy is not available for GPU operations.')
        self.data = cp.array(self.data,dtype=self.data.dtype)
        self.device= 'gpu'
        return self
    def cpu(self)->'Tensor':
        if self.device == 'cpu':return self
        self.data = self.data.get()
        self.device='cpu'
        return self
    def detach(self)->'Tensor':return Tensor(self.data,requires_grad=False,device=self.device,dtype=self.data.dtype)
    def zero_grad(self)->None:
        if self.grad is not None:
            self.grad = None
    def __repr__(self)->str:
        grad_fn = ' ,grad_fn<' + self._ctx.func.__name__ + 'Backward' + str(self.device)+'>' if self._ctx is not None and self.requires_grad else ''
        dtype = f' ,dtype=elysium.{self.data.dtype}'
        if self.data is None:return str(None)
        s = np.array2string(self.data, separator=', ', precision=4).replace('\n', '\n' + ' ' * 7)
        return f"tensor({s}{grad_fn}{dtype})"
    def numpy(self)->np.ndarray:
        if self.requires_grad:raise RuntimeError(f"Can't call numpy() on a Tensor that requires_grad ,Use tensor.detach().numpy()")
        if self.device == 'gpu':
            return self.data.get()
        return self.data
    def backward(self,grad:Optional['Tensor']=None)->None:
        topo_order,visited = [],set()
        def build_topo(t:'Tensor'):
            if t not in visited:
                visited.add(t)
                if t._ctx is not None:
                    for inp in t._ctx.get_saved_tensors():
                        build_topo(inp)
                topo_order.append(t)
        build_topo(self)
        if grad is None:
            assert self.shape == (),f'when no gradient is provided ,backward must be called on a scalar tensor'
            grad = Tensor(1.0,requires_grad=False,device=self.device,dtype=self.dtype)
        assert self.shape == grad.shape, f"grad shape must match tensor shape, {grad.shape!r} != {self.shape!r}"
        self.grad = grad
        for t in reversed(topo_order):
            if t._ctx is not None and t.requires_grad:
                gradient = [t.grad]
                grad_inputs = t._ctx.func.backward(t._ctx,*gradient)
                for i,grad in enumerate(grad_inputs):
                    if grad is not None:
                        if t._ctx.saved_tensors[i].grad is None:
                            t._ctx.saved_tensors[i].grad = grad
                        else:
                            t._ctx.saved_tensors[i].grad.data +=grad.data
                del t._ctx
    def size(self,dim:Optional[int]=None):return self.shape if dim is None else self.shape[dim]
    def sqrt(self)->'Tensor':return Sqrt.apply(self)
    def exp(self)->'Tensor':return Exp.apply(self)
    def cos(self)->'Tensor':return Cos.apply(self)
    def sin(self)->'Tensor':return Sin.apply(self)
    def tan(self)->'Tensor':return Tan.apply(self)
    def log(self)->'Tensor':return Log.apply(self)
    def max(self,axis:Union[Tuple[int,...],int,None]=None,keepdims:Optional[bool]=False)->'Tensor':return Max.apply(self,axis=axis,keepdims=keepdims)
    def min(self,axis:Union[Tuple[int,...],int,None]=None,keepdims:Optional[bool]=False)->'Tensor':return -((-self).max(axis=axis,keepdims=keepdims))
    def sum(self,axis:Union[Tuple[int,...],int,None]=None,keepdim:Optional[bool]=False)->'Tensor':return Sum.apply(self,axis=axis,keepdim=keepdim)
    def mean(self,axis:Union[Tuple[int,...],int,None]=None,keepdim:Optional[bool]=False)->'Tensor':return Mean.apply(self,axis=axis,keepdims=keepdim)
    def var(self,dim:Union[Tuple[int,...],int,None]=None,correction=1,keepdim=False)->'Tensor':return Var.apply(self,dim=dim,correction=correction,keepdim=keepdim)
    def view(self,shape):return View.apply(self,shape)
    def reshape(self,shape:Tuple[int,...])->'Tensor':return Reshape.apply(self,shape)
    def squeeze(self,dim:Union[Tuple[int,...],None]=None)->'Tensor':return Squeeze.apply(self,dim=dim)
    def unsqueeze(self,dim:Tuple[int,...])->'Tensor':return Unsqueeze.apply(self,dim)
    def transpose(self,dim0:int,dim1:int)->'Tensor':return Transpose.apply(self,dim0,dim1)
    def permute(self,dims:Tuple[int,...])->'Tensor':return Permute.apply(self,dims)
    def expand(self,shape)->'Tensor':return Expand.apply(self,shape)
    def repeat_interleave(self,repeats:Union[int,List[int]],axis=None)->'Tensor':return Repeat_Interleave.apply(self,repeats,axis=axis)
    def repeat(self,reps)->'Tensor':return Repeat.apply(self,reps)
    def masked_fill(self,mask:'Tensor',val:float)->'Tensor':return MaskedFill.apply(self,mask,val)
    def tril(self,diagonal=0):return Tril.apply(self,diagonal=diagonal)
    def triu(self,diagonal=0):return Triu.apply(self,diagonal=diagonal)
    @staticmethod
    def cat(ts,axis=0)->'Tensor':return Concat.apply(ts,axis=axis)
    def __getitem__(self,key)->'Tensor':
        xp = cp if self.device == 'gpu' else np
        key=(key,) if not isinstance(key,tuple) else key
        key = tuple(list((i.to(self.device).data.astype(xp.int32) if isinstance(i,Tensor) else i  for i in key)))
        return Index.apply(self,key)
    def __hash__(self):return id(self)
    def __len__(self):return self.shape[0]
    def __neg__(self)->'Tensor':return Neg.apply(self)
    def __add__(self,other:'Tensor')->'Tensor':
        if not isinstance(other,Tensor):other=Tensor(other,device=self.device,dtype=self.dtype)
        return Add.apply(self,other)
    def __iadd__(self,other:'Tesor')->'Tensor':
        if not isinstance(other,Tensor):other=Tensor(other,device=self.device,dtype=self.dtype)
        return Add.apply(self,other,inplace=True)
        
    __radd__ = __add__
    add = __add__
    add_ = __iadd__
    def __sub__(self,other:'Tensor')->'Tensor':
        if not isinstance(other,Tensor):other=Tensor(other,device=self.device,dtype=self.dtype)
        return Sub.apply(self,other)
    def __isub__(self,other:'Tensor')->'Tensor':
        if not isinstance(other,Tensor):other=Tensor(other,device=self.device,dtype=self.dtype)
        return Sub.apply(self,other,inplace=True)
        
    def __rsub__(self,other:'Tensor')->'Tensor':
        if not isinstance(other,Tensor):other=Tensor(other,device=self.device,dtype=self.dtype)
        return sub(other,self)
    sub = __sub__
    sub_ = __isub__
    def __mul__(self,other:'Tensor')->'Tensor':
        if not isinstance(other,Tensor):other=Tensor(other,device=self.device,dtype=self.dtype)
        return Mul.apply(self,other)
    def __imul__(self,other:'Tensor')->'Tensor':
        if not isinstance(other,Tensor):other=Tensor(other,device=self.device,dtype=self.dtype)
        return Mul.apply(self,other,inplace=True)
    def __rmul__(self,other):
        if not isinstance(other,Tensor):other=Tensor(other,device=self.device,dtype=self.dtype)
        return Mul.apply(other,self)
    mul_ = __imul__
    mul = __mul__
    def __truediv__(self,other)->'Tensor':
        if not isinstance(other,Tensor):other=Tensor(other,device=self.device,dtype=self.dtype)
        return Div.apply(self,other)
    def __idiv__(self,other:'Tensor')->'Tensor':
        if not isinstance(other,Tensor):other=Tensor(other,device=self.device,dtype=self.dtype)
        return Div.apply(self,other,inplace=True)
    def _rtruediv__(self,other:'Tensor')->'Tensor':
        if not isinstance(other,Tensor):other=Tensor(other,device=self.device,dtype=self.dtype)
        return dive(other,self)
    div = __truediv__
    div_ = __idiv__
    def __pow__(self,other:'Tensor')->'Tensor':
        if not isinstance(other,Tensor):other=Tensor(other,device=self.device,dtype=self.dtype)
        return Pow.apply(self,other)
    def __ipow__(self,other:'Tensor')->'Tensor':
        if not isinstance(other,Tensor):other=Tensor(other,device=self.device,dtype=self.dtype)
        return Pow.apply(self,other,inplace=True)
    def __rpow__(self,other:'Tensor')->'Tensor':
        if not isinstance(other,Tensor):other=Tensor(other,device=self.device,dtype=self.dtype)
        return Pow.apply(other,self)
    pow = __pow__
    pow_= __ipow__
    def __matmul__(self,other:'Tensor')->'Tensor':
        if self.ndim == 0 or other.ndim==0:raise RuntimeError
        if self.ndim==2 and other.ndim==2:
            return Mm.apply(self,other)
        elif self.ndim == 2 and other.ndim == 1:
            return Mv.apply(self,other)
        elif self.ndim == 1 and other.ndim == 2:
            return Squeeze.apply(Mm.apply(Unsqueeze.apply(self,0),other),0)
        elif self.ndim > 2 and other.ndim == 1:
            return Squeeze.apply(Bmm.apply(self,Unsqueeze.apply(other,-1)),-1)
        elif self.ndim == 1 and other.ndim > 2:
            return Squeeze.apply(Bmm.apply(View.apply(self,(1,*self.shape)),other),-2)
        else:
            return Bmm.apply(self,other)
    @staticmethod
    def maximum(x,y,out=None):
        xp = cp if x.device=='gpu' else np
        if out is not None:return Tensor(xp.maximum(x.data,y.data,out=out),device=x.device,dtype=x.dtype)
        out = xp.maximum(x.data,y.data) 
        return Tensor(out,device=x.device,dtype=x.dtype)

