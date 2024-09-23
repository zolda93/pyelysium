from .functional import pad
class ConstantPad2d:
    def __init__(self,padding,value):
        self.padding = padding
        self.value = value
    def __call__(self,x):return pad(x,self.padding,value=self.value)
class ZeroPad2d(ConstantPad2d):
    def __init__(self,padding):
        self.padding = padding
        self.value = 0.
class ReflectionPad2d:
    def __init__(self,padding):
        self.padding=padding
    def __call__(self,x):return pad(x,self.padding,mode='reflect')
class CircularPad2d:
    def __init__(self,padding):
        self.padding=padding
    def __call__(self,x):return pad(x,self.padding,mode='circular')
class ReplicationPad2d:
    def __init__(self,padding):
        self.padding=padding
    def __call__(self,x):return pad(x,self.padding,mode='replicate')
