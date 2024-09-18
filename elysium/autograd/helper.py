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
    #axis = xp.core.numeric.normalize_axis_tuple(axis,x.ndim)# Convert negative axes to positive indices cupy doesnt have core!!
    axis = tuple(ax if ax >= 0 else x.ndim + ax for ax in axis)
    # Compute the shape of the output array
    out_shape = list(x.shape)
    for ax, win in zip(axis, window_shape):out_shape[ax] = x.shape[ax] - win + 1
    # Full output shape: windowed dimensions come immediately after their respective axes
    full_shape = tuple(out_shape[ax] if ax not in axis else out_shape[ax] for ax in range(x.ndim)) + tuple(window_shape)
    out_strides = tuple(x.strides[ax] for ax in range(x.ndim)) + tuple(x.strides[ax] for ax in axis)# Compute the strides for the output array
    return xp.lib.stride_tricks.as_strided(x, shape=full_shape, strides=out_strides)

def conv2d(x,weight,bias=None,stride=1,padding=0,dilation=1,groups=1,padding_mode='zeros',is_backward_w=False):
    xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
    if is_backward_w:_,c_out,kh,kw=weight.shape
    else:c_out,c_in_by_groups,kh,kw=weight.shape
    kernel_size=(kh,kw)
    if padding != (0,0):
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
    if is_backward_w:
        weight = weight.reshape(n, groups, c_out_group, 1, 1, kh * kw)
        windows = sliding_window_view(x.reshape(n,groups,c_in_group,h,w), kernel_shape, axis=(-3, -2, -1))[:, :, :, ::sh, ::sw, :, ::dh, ::dw]
        h_out, w_out = windows.shape[3:5]
        windows = windows.reshape(n, groups, h_out, w_out, c_in_group , kh * kw)
        if groups == 1:
            y = xp.tensordot(windows, weight, axes=([0,1,5], [0,1,5])).transpose(3,2,0,1,4,5).reshape(c_out,c_in//groups,h_out,w_out)
        else:
            y = xp.einsum("nghwcf,ngxlmf->gxchw", windows, weight,optimize=True).reshape(c_out,c_in//groups,h_out,w_out)
        return y,x
    else:
        weight=weight.reshape(1, groups, c_out_group, 1, 1, c_in_group * kh * kw)
        windows = sliding_window_view(x.reshape(n,groups,c_in_group,h,w), kernel_shape, axis=(-3, -2, -1))[:, :, :, ::sh, ::sw, :, ::dh, ::dw]
        h_out, w_out = windows.shape[3:5]
        windows = windows.reshape(n, groups, 1, h_out, w_out, c_in_group * kh * kw)
        if groups == 1:
            y = xp.tensordot(windows, weight, axes=([1,5], [0, 5])).reshape(n, c_out, h_out, w_out)
        else:
            y = xp.einsum("abcdei,abcdei->abcde", windows, weight,optimize=True).reshape(n, c_out, h_out, w_out)
    if bias is not None:y = y + bias.reshape(1, c_out, 1, 1)
    return y,x,padding
def conv_transpose2d(x,weight,bias=None,stride=1,padding=0,dilation=1,groups=1,output_padding=(0, 0),padding_mode='zeros',input=None,extra_padding=((0,0),(0,0))):
    xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
    x_dilated=dilate(x,stride)
    w_t = xp.flip(weight,axis=(-1,-2))
    c_in,c_out_by_groups,wh,ww=w_t.shape
    n,_,h,w=x.shape
    dilated_kh = (wh - 1) * dilation[0] + 1
    dilated_kw = (ww - 1) * dilation[1] + 1
    x_padded = pad2d(x_dilated, ((wh - 1) * dilation[0], (ww - 1) * dilation[1]))
    x_padded = x_padded.reshape(n, groups, c_in // groups, x_padded.shape[-2], x_padded.shape[-1])
    w_t = w_t.reshape(groups,c_in // groups,c_out_by_groups,wh,ww).transpose(0, 2, 1, 3, 4).reshape(1, groups, c_out_by_groups, 1, 1, (c_in // groups) * wh * ww)
    kernel_shape = (c_in // groups, dilated_kh, dilated_kw)
    windows = sliding_window_view(x_padded, kernel_shape, axis=(-3, -2, -1))[:, :, :, :, :, :, ::dilation[0], ::dilation[1]]
    h_out, w_out = windows.shape[3:5]
    windows = windows.reshape(n, groups, 1, h_out, w_out, (c_in // groups) * wh * ww)
    y = xp.einsum("abcdei,abcdei->abcde", windows, w_t).reshape(n, -1, h_out, w_out)
    if input is not None:Hx,Wx=input.shape[-2:]
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
        if top>0:y[...,top,:] += xp.sum(y[:, :, :top,:],axis=2)
        if bottom>0:y[...,-bottom-1,: ]+=xp.sum(y[:, :, -bottom:,:],axis=2)
        if left>0:y[...,left]+=xp.sum(y[:, :, :, :left],axis=3)
        if right>0:y[...,-right-1]+=xp.sum(y[:, :, :, -right:],axis=3)
    if padding=='same':
        left, right, top, bottom = convert_padding(extra_padding)
        return y[...,top:y.shape[-2] - bottom,left:y.shape[-1] - right]
    y = y[..., padding[0]:Hx + padding[0], padding[1]:Wx + padding[1]]
    # Adjust the output size with padding if necessary
    hig, wig = y.shape[-2:]
    y = xp.pad(y, ((0, 0), (0, 0), (0, Hx - hig), (0, Wx - wig)))
    if bias is not None:y = y + bias.reshape(1, groups * c_out_by_groups, 1, 1)
    return y
def conv2d_backward_w(x,grad, stride, padding, dilation, groups, weight,padding_mode='zeros',is_transpose=False):
    hw, ww = weight.shape[-2:]
    if is_transpose:
        if padding != (0,0):
            if padding == 'same':x ,padding= pad2d(x, padding,stride=stride,kernel_size=(hw,ww),dilation=dilation,padding_mode=padding_mode)
            else:x = pad2d(x,padding,padding_mode=padding_mode)
    h_out, w_out = grad.shape[-2:]
    h_valid = (h_out - 1) * stride[0] + 1 + dilation[0] * (hw - 1)
    w_valid = (w_out - 1) * stride[1] + 1 + dilation[1] * (ww - 1)
    return conv2d( x[..., :h_valid, :w_valid], grad, stride=dilation, padding=(0, 0), dilation=stride,
                   groups=groups,padding_mode=padding_mode,is_backward_w=True)
# pooling helper
def sliding_window_view_pool(x,kernel_size,stride,dilation,padding=(0,0),ceil_mode=False,val=0):
    xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
    import math
    aug = False
    c,h,w = x.shape[-3:]
    wh, ww = kernel_size
    sh, sw = stride
    ph,pw = padding
    dh,dw = dilation
    fn = math.ceil if ceil_mode else math.floor
    h_out = fn((h + 2 * ph - dh * (wh - 1) - 1) / sh + 1)
    w_out = fn((w + 2 * pw - dw * (ww - 1) - 1) / sw + 1)
    if ceil_mode:
        he = (h_out - 1) * sh - (h + 2 * ph - dh * (wh - 1) - 1)
        we = (w_out - 1) * sw - (w + 2 * pw - dw * (ww - 1) - 1)
        if he + ph >= (wh - 1)*dh + 1:
            aug = True
            h_out -= 1
        if we + pw >= (ww - 1)*dw + 1:
            aug = True
            w_out -= 1
    else:
        he,we = 0,0
    pad_width = [[0,0] for _ in range(x.ndim)]
    if aug:
        pad_width[-2] = [ph, 0]
        pad_width[-1] = [pw, 0]
    else:
        pad_width[-2] = [ph, ph + he]
        pad_width[-1] = [pw, pw + we]
    
    padded = xp.pad(x, pad_width, constant_values=val)
    shape = (*x.shape[:-3], h_out, w_out, padded.shape[-3], wh, ww)
    strides = (*padded.strides[:-3],  # batch
               padded.strides[-2] * stride[0],  # H dimension
               padded.strides[-1] * stride[1],  # W dimension
               padded.strides[-3],  # input channel
               padded.strides[-2] * dilation[0],  # kernel height
               padded.strides[-1] * dilation[1],  # kernel width
              )

    windows = xp.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return windows, padded
def maxpool2d(x,kernel_size,stride,dilation,padding,ceil_mode):
    xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
    low_dim = False
    if x.ndim == 3:
        low_dim = True
        x = x[None]
    windows = sliding_window_view_pool(x,kernel_size,stride,dilation,padding,ceil_mode,val=np.inf)[0]
    batch_size,h,w,channels,kh,kw = windows.shape
    max_pooled = xp.max(windows, axis=(4, 5)).transpose(0,3,1,2)
    max_positions = xp.argmax(windows.reshape(batch_size, channels, h,w, -1), axis=-1)
    max_positions = xp.unravel_index(max_positions, (kernel_size[0], kernel_size[1]))
    if low_dim:max_pooled = max_pooled[0]
    return max_pooled, max_positions
def maxpool2d_backward(x, grad, pos, kernel_size, stride, padding, dilation, ceil_mode):
    xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
    low_dim_flag = False
    if x.ndim == 3:
        low_dim_flag = True
        x = x[None]
    x_grad = xp.zeros_like(x)
    idx2, idx1_m = pos
    expanded, padded = sliding_window_view_pool(x_grad, kernel_size, stride, dilation, padding, ceil_mode, val=np.NINF)
    ax0, ax1, ax2, ax3 = xp.indices((expanded.shape[:-2]))
    expanded[ax0, ax1, ax2, ax3, idx2, idx1_m] = xp.moveaxis(grad, -3, -1)
    hp, wp = padding
    h, w = x_grad.shape[-2:]
    x_grad = padded[..., hp:(hp + h), wp:(wp + w)]
    if low_dim_flag:x_grad = x_grad[0]
    return x_grad
def avgpool2d_forward(x, kernel_size, stride=None, padding=(0, 0), ceil_mode=False, count_include_pad=True, divisor_override=None):
    xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
    windows =sliding_window_view_pool(x,kernel_size,stride,(1,1),padding,ceil_mode,val=0)[0]
    batch_size,h,w,channels,kh,kw = windows.shape
    if count_include_pad:
        pool_area = kernel_size[0] * kernel_size[1]
    else:
        pool_area = xp.sum(windows != 0, axis=(4, 5), keepdims=False)
    if divisor_override:
        divisor = divisor_override
    else:
        divisor = pool_area
    out = xp.sum(windows, axis=(4, 5)) / divisor
    return out.transpose(0,3,1,2),divisor
def avgpool2d_backward(x,grad,divisor,kernel_size,stride,padding,ceil_mode=False):
    xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
    x_grad = np.zeros_like(x)
    windows ,padded = sliding_window_view_pool(x_grad,kernel_size,stride,(1,1),padding,ceil_mode,val=0)
    grad_t = grad.transpose(0, 2, 3, 1)
    grad_t /= divisor # Shape: (batch_size, out_h, out_w, channels)
    windows += grad_t[...,None,None]
    hp, wp = padding
    h, w = x_grad.shape[-2:]
    x_grad = padded[..., hp:(hp + h), wp:(wp + w)]
    return x_grad
