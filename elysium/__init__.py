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
def _get_array_module(device):
    """Returns the appropriate array module (NumPy or CuPy) based on the device."""
    return cp if (cp is not None and device == 'gpu') else np

def _create_tensor_from_data(data, requires_grad=False, device='cpu', dtype=np.float32):
    """Creates a Tensor from data, handling the case where data is already a Tensor."""
    if isinstance(data, Tensor):
        return Tensor(data.data, requires_grad=requires_grad, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad, device=device, dtype=dtype)

def tensor(data, requires_grad=False, device='cpu', dtype=np.float32):
    """Creates a Tensor from various data types."""
    return _create_tensor_from_data(data, requires_grad, device, dtype)

def zeros(*shape, requires_grad=False, device='cpu', dtype=np.float32):
    """Creates a Tensor filled with zeros."""
    xp = _get_array_module(device)
    return Tensor(xp.zeros(*shape, dtype=dtype), requires_grad=requires_grad, device=device, dtype=dtype)

def ones(*shape, requires_grad=False, device='cpu', dtype=np.float32):
    """Creates a Tensor filled with ones."""
    xp = _get_array_module(device)
    return Tensor(xp.ones(*shape, dtype=dtype), requires_grad=requires_grad, device=device, dtype=dtype)

def empty(*shape, requires_grad=False, device='cpu', dtype=np.float32):
    """Creates a Tensor with uninitialized values."""
    xp = _get_array_module(device)
    return Tensor(xp.empty(*shape, dtype=dtype), requires_grad=requires_grad, device=device, dtype=dtype)

def rand(*shape, requires_grad=False, device='cpu', dtype=np.float32):
    """Creates a Tensor with values sampled from a uniform distribution [0, 1)."""
    xp = _get_array_module(device)
    return Tensor(xp.random.rand(*shape, dtype=dtype), requires_grad=requires_grad, device=device, dtype=dtype)

def randn(shape, requires_grad=False, device='cpu', dtype=np.float32):
    """Creates a Tensor with values sampled from a standard normal distribution."""
    xp = _get_array_module(device)
    return Tensor(xp.random.default_rng().standard_normal(size=shape, dtype=dtype), requires_grad=requires_grad, device=device, dtype=dtype)

def randint(low, high=None, size=None, requires_grad=False, device='cpu', dtype=np.int32):
    """Creates a Tensor with random integers."""
    xp = _get_array_module(device)
    return Tensor(xp.random.randint(low, high=high, size=size, dtype=dtype), requires_grad=requires_grad, device=device, dtype=dtype)

def arange(start, stop, step=1, requires_grad=False, device='cpu', dtype=np.float32): 
    """Creates a Tensor with values from a given range."""
    xp = _get_array_module(device)
    return Tensor(xp.arange(start, stop, step, dtype=dtype), requires_grad=requires_grad, device=device, dtype=dtype)

# Functions to create tensors like a given tensor
def zeros_like(t, requires_grad=False,device='cpu',dtype=np.float32):
    """Creates a tensor of zeros with the same shape as the input tensor."""
    return zeros(t.shape, requires_grad=requires_grad, device=device,dtype=dtype)

def ones_like(t, requires_grad=False, **kwargs):
    """Creates a tensor of ones with the same shape as the input tensor."""
    return ones(t.shape, requires_grad=requires_grad, **kwargs)

def empty_like(t, requires_grad=False, **kwargs):
    """Creates a tensor with uninitialized values and the same shape as the input tensor."""
    return empty(*t.shape, requires_grad=requires_grad, **kwargs)

def randn_like(t, requires_grad=False, **kwargs):
    """Creates a tensor with values from a standard normal distribution with the same shape as the input tensor."""
    return randn(t.shape, requires_grad=requires_grad, **kwargs)

