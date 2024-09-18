from elysium import *

class Parameter(Tensor):
    def __init__(self,data=None,requires_grad=True,device='cpu',dtype=np.float32):
        if isinstance(data,Tensor):
            data = data.data
        else:
            if data is not None:
                raise TypeError(f"input must be tensor,not {data.__class__.__name__}")
        super().__init__(data,requires_grad=requires_grad,device=device,dtype=dtype)
    def __repr__(self):
        return "Parameter containing:\n"+ super().__repr__()
