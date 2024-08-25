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
                #print(f"GPU {gpu_id}: Memory pool limit set to {available_memory_bytes / (1024**3):.2f} GB")

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



