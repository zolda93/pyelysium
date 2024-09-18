# ! git clone https://github.com/zolda93/pyelysium.git
#%cd /content/pyelysium

from elysium import Tensor,np,cp
from elysium.nn.functional import conv2d,conv_transpose2d

def test_conv2d(n, c_in, c_out, h, w, groups, stride, padding, dilation, kernel_size, padding_mode="zeros"):

    import torch
    import torch.nn as nn

    input = torch.rand(n, c_in, h, w)
    input.requires_grad =True
    conv = nn.Conv2d(
        c_in,
        c_out,
        kernel_size=kernel_size,
        bias=True,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode)

    torch_output = conv(input)
    grad = torch.randn(torch_output.shape)
    torch_output.backward(grad)

    grad_x = input.grad.detach().numpy()
    grad_w = conv.weight.grad
    grad_b = conv.bias.grad
    torch_output = torch_output.detach().numpy()

    input_np = Tensor(input.detach().numpy(),requires_grad=True)
    weight = Tensor(conv.weight.detach().numpy(),requires_grad=True)
    bias = Tensor(conv.bias.detach().cpu().numpy(),requires_grad=True)
    grad = Tensor(grad.detach().numpy())
    output = conv2d(
        input_np,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode)
    output.backward(grad)

    if not np.allclose(output.data, torch_output, atol=1e-4):
        print("Mismatch found forward between PyTorch and custom implementation.")
        break
    if not np.allclose(input.grad.data, grad_x, atol=1e-4):
        print("Mismatch backward input found between PyTorch and custom implementation.")
        break
    if not np.allclose(weight.grad.data, grad_w, atol=1e-4):
        print("Mismatch backward found weight between PyTorch and custom implementation.")
        break
    if not np.allclose(bias.grad.data, grad_b, atol=1e-4):
        print("Mismatch backward found bias between PyTorch and custom implementation.")
        break



def tests():

    # Test padding="same"
    for w in range(40, 60):
        for kernel_size in [1, 2, 3, 4, 5]:
            for dilation in [1, 2, 3, 4]:
                test_conv2d(n=1, c_in=1, c_out=1, h=1, w=w, groups=1, stride=1, padding="same", dilation=dilation, kernel_size=kernel_size)
    # Test 100 random parameters
    for _ in range(200):
        n = np.random.randint(2, 3)
        w = np.random.randint(10, 20)
        h = np.random.randint(10, 20)
        groups = np.random.randint(1, 4)
        c_in = groups * np.random.randint(1, 2)
        c_out = groups * np.random.randint(1, 2)
        stride = (np.random.randint(1, 4), np.random.randint(1, 4))
        padding = (np.random.randint(1, 4), np.random.randint(1, 4))
        dilation = (np.random.randint(1, 4), np.random.randint(1, 4))
        kernel_size = (np.random.randint(1, 4), np.random.randint(1, 4))
        padding_mode = np.random.choice(["zeros", "reflect", "replicate",'circular'])

        test_conv2d(n, c_in, c_out, h, w, groups, stride, padding, dilation, kernel_size,padding_mode=padding_mode)

    print("Tests passed for conv2d!")
tests()


import torch
import torch.nn as nn
for i in range(100):
    batch_size = int(np.random.randint(2, 4))
    groups = int(np.random.randint(1, 5))
    c_in = groups * int(np.random.randint(1, 2))
    c_out = groups * int(np.random.randint(1, 2))
    height = int(np.random.randint(20,30))
    width = int(np.random.randint(20,30))
    kernel_size = (int(np.random.randint(2, 3)), int(np.random.randint(2, 3)))
    stride = (int(np.random.randint(1,2)), int(np.random.randint(1, 2)))
    padding = (int(np.random.randint(0, 2)), int(np.random.randint(0, 2)))
    dilation = (int(np.random.randint(1, 2)), int(np.random.randint(1, 2)))
    output_padding = (int(np.random.randint(0, stride[0])), int(np.random.randint(0, stride[1])))


    input_tensor = torch.randn(batch_size, c_in, height, width, dtype=torch.float32, device='cpu',requires_grad=True)
    trans = nn.ConvTranspose2d(c_in,c_out,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=True,output_padding=output_padding)
    torch_output = trans(input_tensor)
    grad = torch.randn(torch_output.shape)
    torch_output.backward(grad)
    grad_x = input_tensor.grad.detach().numpy()
    grad_w = trans.weight.grad.detach().numpy()
    grad_b = trans.bias.grad.detach().numpy()
    torch_output = torch_output.detach().numpy()
    input = Tensor(input_tensor.detach().numpy(),requires_grad=True)
    weight = Tensor(trans.weight.detach().numpy(),requires_grad=True)
    bias = Tensor(trans.bias.detach().numpy(),requires_grad=True)
    grad = Tensor(grad.detach().numpy())
    y = conv_transpose2d(input, weight, bias=bias,stride=stride, padding=padding, dilation=dilation, groups=groups, output_padding=output_padding)
    y.backward(grad)
    if not np.allclose(y.data, torch_output, atol=1e-4):
        print("Mismatch found forward between PyTorch and custom implementation.")
        break
    if not np.allclose(weight.grad.data, grad_w, atol=1e-4):
        print("Mismatch found between w_grad PyTorch and custom implementation.")
        break
    if not np.allclose(input.grad.data, grad_x, atol=1e-4):
        print("Mismatch found between  x_grad PyTorch and custom implementation.")
        break
    if not np.allclose(bias.grad.data, grad_b, atol=1e-4):
        print("Mismatch backward found weight between PyTorch and custom implementation.")
        break

print('test passed for conv_transpose2d !')
