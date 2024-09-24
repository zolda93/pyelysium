from elysium import Tensor,zeros,ones,cp,np
from .functional import batch_norm_impl
from . import init
from .parameter import Parameter
class BatchNorm2d:
    def __init__(self,num_features,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True,training=True,device='cpu'):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.training = training
        self.weight,self.bias = (Parameter(ones(num_features)).to(device),Parameter(zeros(num_features)).to(device)) if self.affine else (None,None)
        self.running_mean,self.running_var,self.num_batches_tracked = (zeros(num_features).to(device),ones(num_features).to(device),Tensor(0,dtype=np.int32).to(device)) if self.track_running_stats else (None,None,None)
        self.reset_parameters()
    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean = init.zeros_(self.running_mean)
            self.running_var = init.ones_(self.running_var)
            self.num_batches_tracked = init.zeros_(self.num_batches_tracked)
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
    def _check_input_dim(self,x):
        if x.ndim != 4:raise ValueError('expected 4D input (got {}D input)'.format(x.ndim))
    def __call__(self,x):
        self._check_input_dim(x)
        if self.training and self.track_running_stats:
            self.num_batches_tracked.data += 1
            if self.momentum is None:  
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        else:
            exponential_average_factor = None
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        out,self.running_mean,self.running_var=batch_norm_impl(x,self.running_mean if not self.training or self.track_running_stats else None,
                            self.running_var if not self.training or self.track_running_stats else None,
                            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
        return out



