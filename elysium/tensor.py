from typing import Tuple,List,Union,Optional,TypeVar
from elysium import np,cp,ConstType,Dtype,TensorType
from .autograd.function import *
T = TypeVar('T',bound='Tensor')

class Tensor:
    __slots__ = ('data', '_requires_grad', 'grad','_ctx', 'device')
    __deletable__ = ('_ctx',)
    def __init__(self,
            data:TensorType,
            requires_grad:Optional[bool]=False,
            device:Optional[Union[str,tuple,list]]='cpu',
            dtype:Optional[Dtype] = np.float32)->None:
        self._requires_grad = requires_grad
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
        elif isinstance(data,(np.ndarray,cp.ndarray)):
            return data.astype(dtype)
        else:
            raise TypeError(f"Unsupported data type for Tensor initialization: {type(data)}")

    def __del__(self):
        try:
            if self._ctx is not None:
                self._ctx = None
        except Exception as e:
            print(f"Exception in __del__ of Tensor: {e}")

    @property
    def requires_grad(self)->bool:return self._requires_grad
    @requires_grad.setter
    def requires_grad(self,value:bool)->None:
        if not self._is_floating_point():
            raise ValueError("Cannot require gradients for non-floating-point tensor types.")
        self._requires_grad = value
        if value and self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data), requires_grad=False, device=self.device)
    def is_floating_point(self)->bool:
        return self.data.dtype in [np.float16,np.float32,np.float64]
    @property
    def shape(self)->Tuple[int,...]:return self.data.shape
    @property
    def size(self)->int:return self.data.size
    @property
    def dtype(self)->Dtype:return self.data.dtype
    @property
    def ndim(self)->int:return self.data.ndim
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
        return Tensor(cp.array(self.data),requires_grad=self.requires_grad,device='gpu',dtype=self.data.dtype)
    def cpu(self)->'Tensor':
        if self.device == 'cpu':return self
        return Tensor(self.data.get(),requires_grad=self.requires_grad,device='cpu',dtype=self.data.dtype)
    def detach(self)->'Tensor':return Tensor(self.data,requires_grad=False,device=self.device,dtype=self.data.dtype)
    def zero_grad(self)->None:
        if self.grad is not None:
            self.grad.data.fill(0)
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
    def size(self,dim:Optional[int]=None):return self.shape if dim is None else self.shape[dim]
    def __neg__(self)->'Tensor':return Neg.apply(self)
    def __add__(self,other:'Tensor')->'Tensor':return Add.apply(self,other)
    def __sub__(self,other:'Tensor')->'Tensor':return Add.apply(self,(-other))
    def __mul__(self,other:'Tensor')->'Tensor':return Mul.apply(self,other)


    


        



