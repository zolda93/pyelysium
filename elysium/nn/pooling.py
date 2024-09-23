from .functional import maxpool2d,avg_pool2d,pair
class AvgPool2d:
    def __init__(self,kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride) if stride is not None else kernel_size
        self.padding=pair(padding)
        self.ceil_mode=ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override=divisor_override
    def __call__(self,x):return avg_pool2d(x,self.kernel_size,stride=self.stride,padding=self.padding,ceil_mode=self.ceil_mode,count_include_pad=self.count_include_pad,divisor_override=self.divisor_override)
class MaxPool2d:
    def __init__(self,kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        self.kernel_size=pair(kernel_size)
        self.stride=pair(stride) if stride is not None else kernel_size
        self.padding=pair(padding)
        self.dilation=pair(dilation)
        self.return_indices=return_indices
        self.ceil_mode=ceil_mode
    def __call__(self,x):return maxpool2d(x,self.kernel_size,stride=self.stride,padding=self.padding,dilation=self.dilation,ceil_mode=self.ceil_mode,return_indices=self.return_indices)
