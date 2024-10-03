from elysium import np,cp

# helper function for convolution
def convert_padding(padding):
    """Convert padding to (left, right, top, bottom) format."""
    if isinstance(padding, int):
        return (padding, padding, padding, padding)
    elif isinstance(padding,(list, tuple)) and len(padding) == 2:
        return (padding[1], padding[1], padding[0], padding[0])
    elif isinstance(padding, (list,tuple)) and len(padding) == 4:
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
def calculate_padding(x,padding, stride=None, dilation=None, kernel_size=None):
    if padding == 'same':
        h, w = x.shape[-2:]
        sh, sw = stride if isinstance(stride, (tuple, list)) else (stride, stride)
        dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
        kh, kw = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        ph = (h - 1) * (sh - 1) + (kh - 1) * dh
        pw = (w - 1) * (sw - 1) + (kw - 1) * dw
        ph0, ph1 = ph // 2, ph - ph // 2
        pw0, pw1 = pw // 2, pw - pw // 2
        return  pw0,pw1,ph0,ph1
    return convert_padding(padding)
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
def conv2d(x,weight,bias=None,stride=1,dilation=1,groups=1,is_backward_w=False,transpose=False):
    xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
    assert x.dtype==weight.dtype
    if is_backward_w: _,c_out,kh,kw=weight.shape
    else:c_out,c_in_by_groups,kh,kw=weight.shape
    kernel_size=(kh,kw)
    n,c_in,h,w=x.shape
    dh,dw=dilation
    sh,sw=stride
    dilated_kh,dilated_kw=(kh-1)*dh + 1,(kw-1)*dw + 1
    c_in_group = c_in // groups
    c_out_group = c_out // groups 
    kernel_shape = (c_in_group, dilated_kh, dilated_kw)
    if is_backward_w:
        weight = weight.reshape(n, groups, c_out_group, 1, 1, kh * kw)
        windows = sliding_window_view(x.reshape(n,groups,c_in_group,h,w), kernel_shape, axis=(-3, -2, -1))[:, :, :, ::sh, ::sw, :, ::dh, ::dw]
        h_out, w_out = windows.shape[3:5]
        windows = windows.reshape(n, groups, h_out, w_out, c_in_group , kh * kw)
        y =  xp.einsum("nghwcf,ngxlmf->gxchw", windows, weight,optimize=True).reshape(c_out,c_in_group,h_out,w_out)
        return y
    else:
        if transpose:
            weight = xp.flip(weight,axis=(-1,-2))
            _,_,kh,kw = weight.shape
            weight = weight.reshape(groups,c_in_group,-1,kh,kw).transpose(0, 2, 1, 3, 4).reshape(1, groups,-1, 1, 1, c_in_group * kh * kw)
            x = pad2d(dilate(x,stride), ((kh - 1) * dh, (kw - 1) * dw))
            windows = sliding_window_view(x.reshape(n, groups, c_in // groups, x.shape[-2], x.shape[-1]), kernel_shape, axis=(-3, -2, -1))[...,::dh,::dw]
            h_out, w_out = windows.shape[3:5]
            windows = windows.reshape(n, groups, 1, h_out, w_out, c_in_group * kh * kw)
            y =  xp.einsum("abcdei,abcdei->abcde", windows, weight,optimize=True).reshape(n,-1, h_out, w_out)
            return y
        else:
            weight=weight.reshape(1, groups,-1, 1, 1, c_in_group * kh * kw)
            windows = sliding_window_view(x.reshape(n,groups,c_in_group,h,w), kernel_shape, axis=(-3, -2, -1))[...,::sh, ::sw,:,::dh,::dw]
            h_out, w_out = windows.shape[3:5]
            windows = windows.reshape(n, groups, 1, h_out, w_out, c_in_group * kh * kw)
            y = xp.einsum("abcdei,abcdei->abcde", windows, weight,optimize=True).reshape(n,-1, h_out, w_out)
            return y
def conv2d_backward_w(x,grad, stride,dilation, groups, weight):
    hw, ww = weight.shape[-2:]
    h_out, w_out = grad.shape[-2:]
    h_valid = (h_out - 1) * stride[0] + 1 + dilation[0] * (hw - 1)
    w_valid = (w_out - 1) * stride[1] + 1 + dilation[1] * (ww - 1)
    return conv2d( x[..., :h_valid, :w_valid], grad, stride=dilation,dilation=stride,
                   groups=groups,is_backward_w=True)
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
    strides = (*padded.strides[:-3],  # batch_size
               padded.strides[-2] * stride[0],  # h dimension
               padded.strides[-1] * stride[1],  # w dimension
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
    windows,padded = sliding_window_view_pool(x,kernel_size,stride,dilation,padding,ceil_mode,val=xp.NINF)
    batch_size,h,w,channels,kh,kw = windows.shape
    max_pooled = xp.max(windows, axis=(4, 5)).transpose(0,3,1,2)
    max_positions = xp.argmax(windows.transpose(0,3,1,2,4,5).reshape(batch_size, channels, h,w,-1), axis=-1)
    max_positions = xp.unravel_index(max_positions, (kernel_size[0], kernel_size[1]))
    if low_dim:max_pooled = max_pooled[0]
    return max_pooled, max_positions
def maxpool2d_backward(x, grad, pos, kernel_size, stride, padding, dilation, ceil_mode):
    xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
    if xp is cp:
        import cupyx
        add_at = cupyx.scatter_add
    else:
        add_at = np.add.at
    low_dim_flag = False
    if x.ndim == 3:
        low_dim_flag = True
        x = x[None]
    x_grad = xp.zeros_like(x)
    idx2, idx1_m = pos
    expanded, padded = sliding_window_view_pool(x_grad, kernel_size, stride, dilation, padding, ceil_mode, val=0.0)
    expanded = expanded.transpose(0,3,1,2,4,5)
    ax0, ax1, ax2, ax3 = xp.indices((expanded.shape[:-2]))
    add_at(expanded,(ax0,ax1,ax2,ax3,idx2,idx1_m),grad)
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
def col2im_gpu(col, sy, sx, ph, pw, h, w):
    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img = cp.empty((n, c, h, w),dtype=col.dtype)
    cp.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img
def col2im(xp,col,input_shape,kernel_size,stride,padding):
    N,C,H,W = input_shape
    kh,kw = kernel_size
    sh,sw = stride
    ph,pw = padding
    OH = (H + 2*ph - kh)//sh + 1
    OW = (W + 2*pw - kw)//sw + 1
    if xp == cp:
        x = col2im_gpu(col, sh, sw, ph, pw, H, W)
    else:
        x = np.zeros((N, C, H + 2 * ph + sh - 1, W + 2 * pw + sw - 1),dtype=col.dtype)
        for j in range(kh):
            j_lim = j + sh*OH
            for i in range(kw):
                i_lim = i + sw*OW
                x[:, :, j:j_lim:sh, i:i_lim:sw] += col[:, :, j, i, :, :]
        x = x[:, :, ph:H + ph, pw:W + pw]
    return x    
def avgpool2d_backward(x,grad,divisor,kernel_size,stride,padding,ceil_mode=False):
    xp = cp if (cp is not None and x.__class__ is cp.ndarray) else np
    if isinstance(divisor,xp.ndarray):divisor = divisor.transpose(0,3,1,2)
    N,C,H,W = x.shape
    _,_,h_out,w_out = grad.shape
    kh,kw = kernel_size
    #sh,sw = stride
    #ph,pw = padding
    grad /= divisor
    grad = xp.broadcast_to(grad.reshape(-1),(kh,kw,N*C*h_out*w_out)).reshape(kh,kw,N,C,h_out,w_out).transpose(2,3,0,1,4,5)
    return col2im(xp,grad,x.shape,kernel_size,stride,padding)
