import math
from elysium import zeros,cp,np
from .functional import pair,conv2d,conv_transpose2d
from .parameter import Parameter
from . import init
class Conv:
    def __init__(self,in_channels,out_channels,kernel_size,bias=True,transpose=False,stride=1,padding=0,dilation=1,groups=1,padding_mode='zeros',device='cpu'):
        if in_channels  % groups != 0:raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:raise ValueError("out_channels must be divisible by groups")
        self.kernel_size = pair(kernel_size)
        self.stride=pair(stride)
        self.padding=pair(padding) if not isinstance(padding,str) else padding
        self.dilation=pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        if transpose:
            self.weight = Parameter(zeros((in_channels,out_channels//groups,*self.kernel_size))).to(device)
        else:
            self.weight = Parameter(zeros((out_channels,in_channels//groups,*self.kernel_size))).to(device)
        self.bias = Parameter(zeros(out_channels)).to(device) if bias else None
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            fan_in,_ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
    def to(self,device):
        self.weight = self.weight.to(device)
        if self.bias is not None :self.bias=self.bias.to(device)
        return self
class Conv2d(Conv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device='cpu'):
        super().__init__(in_channels, out_channels, kernel_size, bias=bias, transpose=False, 
                         stride=stride, padding=padding, dilation=dilation, 
                         groups=groups, padding_mode=padding_mode, device=device)

    def __call__(self, x): 
        return conv2d(x, self.weight, bias=self.bias, stride=self.stride, 
                        padding=self.padding, dilation=self.dilation, 
                        groups=self.groups, padding_mode=self.padding_mode)

class ConvTranspose2d(Conv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device='cpu'):
        if padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')
        
        self.output_padding = output_padding
        super().__init__(in_channels, out_channels, kernel_size, bias=bias, transpose=True,
                         stride=stride, padding=padding, dilation=dilation, 
                         groups=groups, padding_mode=padding_mode, device=device)

    def __call__(self, x):
        return conv_transpose2d(x, self.weight, bias=self.bias, stride=self.stride,
                                    padding=self.padding, output_padding=self.output_padding,
                                    groups=self.groups, dilation=self.dilation)





