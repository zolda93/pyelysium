from . import functional as F
class Sigmoid:
    def __call__(self,x):return F.sigmoid(x)
class LogSigmoid:
    def __call__(self,x):return F.logsigmoid(x)
class ReLU:
    def __init__(self,inplace=False):
        self.inplace=inplace
    def __call__(self,x):return F.relu(x,inplace=self.inplace)
class Tanh:
    def __call__(self,x):return F.tanh(x)
class LeakyReLU:
    def __init__(self,negative_slope=0.01, inplace=False):
        self.negative_slope=negative_slope
        self.inplace=inplace
    def __call__(self,x):return F.leaky_relu(x,negative_slope=self.negative_slope,inplace=self.inplace)
class Softmax:
    def __init__(self,dim=None):
        self.dim=dim
    def __call__(self,x):return F.softmax(x,dim=self.dim)
class LogSoftmax:
    def __init__(self,dim=None):
        self.dim=dim
    def __call__(self,x):return F.log_softmax(x,dim=self.dim)
