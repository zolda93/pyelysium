from elysium import np,cp

# helper function for convolution
def dilate(x,dilation):
    xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
    if isinstance(dilation,int):dilation=(dilation,dilation)
    hd,wd = dilation
    hk,wk = x.shape[-2:]
    dilated = xp.zeros((*x.shape[:-2], (hk - 1) * hd + 1, (wk - 1) * wd + 1), dtype=x.dtype)
    dilated[...,::hd,::wd] = x
    return dilated
def pad2d(x, padding, stride=None, dilation=None, kernel_size=None, padding_mode='zeros'):
    xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
    if padding == 'valid':
        return x
    if padding == 'same':
        h, w = x.shape[-2:]
        sh, sw = stride if isinstance(stride, (tuple, list)) else (stride, stride)
        dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
        kh, kw = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        ph = (h - 1) * (sh - 1) + (kh - 1) * dh
        pw = (w - 1) * (sw - 1) + (kw - 1) * dw
        ph0, ph1 = ph // 2, ph - ph // 2
        pw0, pw1 = pw // 2, pw - pw // 2
    else:
        ph0, pw0 = padding if isinstance(padding, (tuple, list)) else (padding, padding)
        ph1, pw1 = ph0, pw0
    pad_width = ((0, 0), (0, 0), (ph0, ph1), (pw0, pw1))
    mode = {
        "zeros": "constant",
        "reflect": "reflect",
        "replicate": "edge",
        "circular": "wrap",
    }.get(padding_mode, 'zeros')
    return xp.pad(x, pad_width,mode)
# cupy doesnt have a sliding_window_view function
def sliding_window_view(x, window_shape, axis=None):
    xp = cp if  (cp is not None and x.__class__ is cp.ndarray) else np
    if isinstance(window_shape, int):window_shape = (window_shape,)  # Convert single int to tuple
    if axis is None:axis = tuple(range(x.ndim)) # Apply sliding window to all axes
    if isinstance(axis,int):axis=(axis,)
    if len(window_shape)!= len(axis):raise ValueError("window_shape must have the same length as axis")
    axis = xp.core.numeric.normalize_axis_tuple(axis,x.ndim)# Convert negative axes to positive indices
    # Compute the shape of the output array
    out_shape = list(x.shape)
    for ax, win in zip(axis, window_shape):out_shape[ax] = x.shape[ax] - win + 1
    # Full output shape: windowed dimensions come immediately after their respective axes
    full_shape = tuple(out_shape[ax] if ax not in axis else out_shape[ax] for ax in range(x.ndim)) + tuple(window_shape)
    out_strides = tuple(x.strides[ax] for ax in range(x.ndim)) + tuple(x.strides[ax] for ax in axis)# Compute the strides for the output array
    return xp.lib.stride_tricks.as_strided(x, shape=full_shape, strides=out_strides)

