from elysium import empty,cp,np
from .functional import group_norm
from . import init
from .parameter import Parameter
class GroupNorm:
    def __init__(self,num_groups, num_channels, eps=1e-05, affine=True,device='cpu'):
        if num_channels % num_groups != 0:raise ValueError('num_channels must be divisible by num_groups')
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.weight,self.bias = (Parameter(empty((num_channels))).to(device),Parameter(empty((num_channels)).to(device))) if self.affine else (None,None)
        self.reset_parameters()
    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
    def __call__(self, x):return group_norm(x, self.num_groups, weight=self.weight, bias=self.bias, eps=self.eps)
