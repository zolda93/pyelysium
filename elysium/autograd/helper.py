from elysium import np,cp

# helper function for convolution
def convert_padding(padding):
    """Convert padding to (left, right, top, bottom) format."""
    if isinstance(padding, int):
        return (padding, padding, padding, padding)
    elif isinstance(padding, tuple) and len(padding) == 2:
        return (padding[1], padding[1], padding[0], padding[0])
    elif isinstance(padding, tuple) and len(padding) == 4:
        return padding
    else:
        raise ValueError("Invalid padding format.")
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
    return (xp.pad(x, pad_width,mode),(ph0, ph1, pw0, pw1) )if padding=='same' else xp.pad(x,pad_width,mode)

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

def conv2d(x,w,bias=None,stride=1,padding=0,dilation=1,groups=1,padding_mode='zeros',is_backward_w=False):
    xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
    if isinstance(stride,int):stride=(stride,stride)
    if isinstance(dilation,int):dilation=(dilation,dilation)
    if isinstance(padding,int):padding=(padding,padding)
    if is_backward_w:_,c_out,kh,kw=w.shape
    else:c_out,c_in_by_groups,kh,kw=w.shape
    kernel_size=(kh,kw)
    if padding:
        if padding=='same':x,padding=pad2d(x,padding,stride,dilation,kernel_size,padding_mode)
        else:x=pad2d(x,padding,stride,dilation,kernel_size,padding_mode)
    n,c_in,h,w=x.shape
    dh,dw=dilation
    sh,sw=stride
    dilated_kh,dilated_kw=(kh-1)*dh + 1,(kw-1)*dw + 1
    assert c_in % groups == 0, f"Number of input channels ({c_in}) not divisible by groups ({groups})."
    assert c_out % groups == 0, f"Number of output channels ({c_out}) not divisible by groups ({groups})."
    c_in_group = c_in // groups
    c_out_group = c_out // groups
    kernel_shape = (c_in_group, dilated_kh, dilated_kw)
    if is_backward_w:w=w.reshape(n, groups, c_out_group, 1, 1, kh * kw)
    else:w=w.reshape(1, groups, c_out_group, 1, 1, c_in_group * kh * kw)
    windows = sliding_window_view(x.reshape(n,groups,c_in_group,h,w), kernel_shape, axis=(-3, -2, -1))[:, :, :, ::sh, ::sw, :, ::dh, ::dw]
    h_out, w_out = windows.shape[3:5]
    if is_backward_w:windows = windows.reshape(n, groups, h_out, w_out, c_in_group , kh * kw)
    else:windows = windows.reshape(n, groups, 1, h_out, w_out, c_in_group * kh * kw)
    if is_backward_w:
        y = xp.einsum("nghwcf,ngxlmf->gxchw", windows, weight).reshape(c_out,c_in//groups,h_out,w_out)
        return y,x
    else:
        y = xp.einsum("abcdei,abcdei->abcde", windows, weight).reshape(n, c_out, h_out, w_out)    
    if bias is not None:y = y + bias.reshape(1, c_out, 1, 1)
    return y,x,padding

def conv_transpose2d(x,w,bias=None,stride=1,padding=0,dilation=1,groups=1,output_padding=(0, 0),padding_mode='zeros',x=None,extra_padding=((0,0),(0,0))):
    xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
    if isinstance(stride,int):stride=(stride,stride)
    if isinstance(dilation,int):dilation=(dilation,dilation)
    x_dilated=dilate(x,stride)
    w_t = xp.flip(w,axis=(-1,-2))
    c_in,c_out_by_groups,wh,ww=w_t.shape
    n,_,h,w=x.shape
    dilated_kh = (wh - 1) * dilation[0] + 1
    dilated_kw = (ww - 1) * dilation[1] + 1
    x_padded = pad2d(x_dilated, ((wh - 1) * dilation[0], (ww - 1) * dilation[1]))
    xpadded = xpadded.reshape(n, groups, c_in // groups, input_padded.shape[-2], input_padded.shape[-1])
    w_t = w_t.reshape(groups,c_in // groups,c_out_by_groups,wh,ww).transpose(0, 2, 1, 3, 4).reshape(1, groups, c_out_by_groups, 1, 1, (c_in // groups) * wh * ww)
    kernel_shape = (c_in // groups, dilated_kh, dilated_kw)
    windows = sliding_window_view(x_padded, kernel_shape, axis=(-3, -2, -1))[:, :, :, :, :, :, ::dilation[0], ::dilation[1]]
    h_out, w_out = windows.shape[3:5]
    windows = windows.reshape(n, groups, 1, h_out, w_out, (c_in // groups) * wh * ww)
    y = xp.einsum("abcdei,abcdei->abcde", windows, weight_t).reshape(n, -1, h_out, w_out)
    if x is not None:Hx,Wx=x.shape[-2:]
    else:
        hop, wop = output_padding if len(output_padding) == 2 else (output_padding[0], output_padding[0])
        Hx = (x.shape[-2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (weight.shape[-2] - 1) + hop + 1
        Wx = (x.shape[-1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (weight.shape[-1] - 1) + wop + 1
    # this part for calculating gradient of the input from a convolution forward operation
    if padding_mode=='reflect':
        left, right, top, bottom = convert_padding(padding)
        h_in = y.shape[2] - top - bottom
        w_in = y.shape[3] - left - right
        y=xp.pad(y, ((0, 0), (0, 0), (0, Hx-h_in if Hx-h_in >0 else 0), (0, Wx-w_in if Wx - w_in >0 else 0)))
        if top > 0 :y[...,top+1:2*top+1,:]         +=xp.flip(y[:, :, :top,:], axis=2)
        if bottom>0:y[...,-2*bottom-1:-bottom-1,:] +=xp.flip(y[:, :, -bottom:,:], axis=2)
        if left>0  :y[...,left+1:2*left+1]         +=xp.flip(y[:, :, :, :left], axis=3)
        if right>0 :y[...,-2*right-1:-right-1]     +=xp.flip(y[:, :, :, -right:], axis=3)
    elif padding_mode=='circular':
        left, right, top, bottom = convert_padding(padding)
        h_in = y.shape[2] - top - bottom
        w_in = y.shape[3] - left - right
        y=xp.pad(y, ((0, 0), (0, 0), (0, Hx-h_in if Hx-h_in >0 else 0), (0, Wx-w_in if Wx - w_in >0 else 0)))
        if top > 0 and bottom > 0:
            y[..., top:top+bottom, :] += y[:, :, -bottom:, :]
            y[..., -bottom - top:-bottom, :] += y[:, :,:top, :]
        elif top > 0 and bottom <= 0:  
            y[..., -top:, :] += y[:, :, :top, :]
        elif bottom > 0 and top <= 0:  
            y[..., :bottom, :] += y[:, :, -bottom:, :]

        if left > 0 and right > 0:
            y[..., -right - left:-right] += y[:, :, :, :left]
            y[..., left:left + right] += y[:, :, :, -right:]
        elif left > 0 and right <= 0:  # Right padding is 0
            y[..., -left:] += y[:, :, :, :left]
        elif right > 0:  # Left padding is 0
            y[..., :right] += y[:, :, :, -right:]
    elif padding_mode=='replicate':
        left, right, top, bottom = convert_padding(padding)
        h_in = y.shape[2] - top - bottom
        w_in = y.shape[3] - left - right
        y=xp.pad(y, ((0, 0), (0, 0), (0, Hx-h_in if Hx-h_in >0 else 0), (0, Wx-w_in if Wx - w_in >0 else 0)))
        if top>0:dout[...,top,:] += xp.sum(y[:, :, :top,:],axis=2)
        if bottom>0:dout[...,-bottom-1,: ]+=xp.sum(y[:, :, -bottom:,:],axis=2)
        if left>0:dout[...,left]+=xp.sum(y[:, :, :, :left],axis=3)
        if right>0:y[...,-right-1]+=xp.sum(y[:, :, :, -right:],axis=3)
    if padding=='same':
        left, right, top, bottom = convert_padding(extra_padding)
        return y[...,top:y.shape[-2] - bottom,left:y.shape[-1] - right]
    y = y[..., padding[0]:Hx + padding[0], padding[1]:Wx + padding[1]]
    # Adjust the output size with padding if necessary
    hig, wig = y.shape[-2:]
    y = xp.pad(y, ((0, 0), (0, 0), (0, Hx - hig), (0, Wx - wig)))
    if bias is not None:y = y + bias.reshape(1, c_out, 1, 1)
    return y

def conv2d_backward_w( x, grad, stride, padding, dilation, groups, weight,padding_mode='zeros'):
    if isinstance(stride,int):stride=(stride,stride)
    if isinstance(dilation,int):dilation=(dilation,dilation)
    if isinstance(padding,int):padding=(padding,padding)
    hw, ww = weight.shape[-2:]
    if padding == 'same':x_padded ,padding= pad2d(x, padding,stride=stride,kernel_size=(hw,ww),dilation=dilation,padding_mode=padding_mode)
    else:x_padded = pad2d(x,padding,padding_mode=padding_mode)
    H_out, W_out = grad.shape[-2:]
    H_valid = (H_out - 1) * stride[0] + 1 + dilation[0] * (hw - 1)
    W_valid = (W_out - 1) * stride[1] + 1 + dilation[1] * (ww - 1)
    return conv2d( x_padded[..., :H_valid, :W_valid], grad, stride=dilation, padding=(0, 0), dilation=stride,
                   groups=groups,padding_mode=padding_mode,is_backward_w=True)
