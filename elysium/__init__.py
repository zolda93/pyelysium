from typing import Union,List
import numpy as np

ConstType = Union[float,int,bool]
Dtype = Union[np.int16,np.int32,np.int64,np.float16,np.float32,np.float64,np.complex64,np.complex128]
TensorType = Union[None,ConstType,List,np.ndarray]

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

if  CUPY_AVAILABLE:
    cp.cuda.Device()
    num_gpus = cp.cuda.runtime.getDeviceCount() # Get the number of available GPUs
    for gpu_id in range(num_gpus):
        with cp.cuda.Device(gpu_id):
            # Get basic memory properties
            props = cp.cuda.runtime.getDeviceProperties(gpu_id)
            total_memory_bytes = props['totalGlobalMem']

            # Calculate available memory (in bytes)
            # Note: There's no direct way to get truly "available" memory.
            # This is an approximation.
            available_memory_bytes = total_memory_bytes

            try:
                # Set memory pool limit for the current device
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(available_memory_bytes)
            

            except Exception as e:
                print(f"GPU {gpu_id}: Error setting memory pool limit: {e}")


class no_grad:
    _enabled = False  # Class-level attribute to track no_grad status

    def __enter__(self):
        self.prev = no_grad._enabled  # Save the previous state
        no_grad._enabled = True  # Disable gradient tracking
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        no_grad._enabled = self.prev  # Restore the previous state

from .tensor import *

def tensor(data,requires_grad=False,**kwargs):
    if isinstance(data,Tensor):return Tensor(data.data,requires_grad=requires_grad,**kwargs)
    return Tensor(data,requires_grad=requires_grad,**kwargs)
def zeros(*shape,requires_grad=False,**kwargs):
    xp = cp  if (cp is not None and kwargs['device'] == 'gpu') else np
    return Tensor(xp.zeros(*shape,dtype=np.float32),requires_grad=requires_grad,**kwargs)
def ones(*shape,requires_grad=False,**kwargs):
    xp = cp  if (cp is not None and kwargs['device'] == 'gpu') else np
    return Tensor(xp.ones(*shape,dtype=np.float32),requires_grad=requires_grad,**kwargs)
def empty(*shape,requires_grad=False,**kwargs):
    xp = cp  if (cp is not None and kwargs['device'] == 'gpu') else np
    return Tensor(xp.empty(*shape,dtype=np.float32),requires_grad=requires_grad,**kwargs)
def rand(*shape,requires_grad=False,**kwargs):
    xp = cp  if (cp is not None and kwargs['device'] == 'gpu') else np
    return Tensor(xp.random.rand(*shape,dtype=np.float32),requires_grad=requires_grad,**kwargs)
def randn(shape,requires_grad=False,**kwargs):
    return Tensor(xp.random.default_rng().standard_normal(size=shape, dtype=np.float32),requires_grad=requires_grad,**kwargs)
def randint(low,high=None,size=None,requires_grad=False,**kwargs):
    return Tensor(xp.random.randint(low,high=high,size=size),requires_grad=requires_grad,**kwargs)
def arange(stop,start=0,requires_grad=False,**kwargs):
    xp = cp  if (cp is not None and kwargs['device'] == 'gpu') else np
    return Tensor(xp.arange(start=start,stop=stop,dtype=np.float32),requires_grad=requiress_grad,**kwargs)
def zeros_like(t,requires_grad=False,**kwargs):return zeros(*t.shape,requires_grad=requires_grad,**kwargs)
def ones_like(t,requires_grad=False,**kwargs):return ones(*t.shape,requires_grad=requires_grad,**kwargs)
def empty_like(t,requires_grad=False,**kwargs):return empty(*t.shape,requires_grad=requires_grad,**kwargs)
def randn_like(t,requires_grad=False,**kwargs):return randn(t.shape,requires_grad=requires_grad,**kwargs)


