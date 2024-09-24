import math
from elysium import zeros,cp,np
from .functional import linear
from .parameter import Parameter
from . import init
class Linear:
    def __init__(self,in_features,out_features,bias=True,device='cpu'):
        self.in_features=in_features
        self.out_features=out_features
        self.weight = Parameter(zeros((out_features,in_features))).to(device)
        self.bias = Parameter(zeros(out_features)).to(device) if bias else None
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            fan_in,_ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1/math.sqrt(fan_in)
            xp = cp if self.bias.device=='gpu' else np
            self.bias.data = xp.random.uniform(low=-bound, high=bound, size=self.bias.shape).astype(xp.float32)
    def __call__(self,x):return linear(x,self.weight,bias=self.bias)
