from elysium import empty
from .functional import layer_norm
from . import init
from .parameter import Parameter
class LayerNorm:
    def __init__(self,normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device='cpu'):
        if isinstance(normalized_shape, int):normalized_shape = (normalized_shape,) 
        self.normalized_shape = tuple(normalized_shape) 
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(empty((self.normalized_shape))).to(device)
            if bias:
                self.bias = Parameter(empty((self.normalized_shape))).to(device)
            else:
                self.bias = None
        else:
            self.weight = None
            self.bias=None
        self.reset_parameters()
    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)
    def __call__(self,x):return layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)
