from elysium import *

class Parameter(Tensor):
    def __init__(self,t=None,device='cpu',dtype=np.float32):
        if not isinstance(t,Tensor) or t is None:raise TypeError(f"input must be tensor,not {t.__class__.__name__}")
        super().__init__(t.data,requires_grad=True,device=t.device,dtype=t.dtype)
    def __repr__(self):
        return "Parameter containing:\n"+ super().__repr__()
