import torch
from torch.autograd import Function


class MultiplyAdd(Function):
    @staticmethod
    def forward(ctx, w, x, b):
        ctx.save_for_backward(w, x)
        output = w * x + b
        return output

    @staticmethod
    # 注意，这里的求导数并不是只求该操作的output对变量的导数，而是指最终输出对该变量的导数
    def backward(ctx, grad_output):
        w, x = ctx.saved_variables
        grad_w = grad_output * x
        grad_x = grad_output * w
        grad_b = grad_output * 1
        # backward的输出参数和forward的输出参数必须一一对应
        return grad_w, grad_x, grad_b


x = torch.ones(1, requires_grad=True)
w = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

y = MultiplyAdd.apply(w, x, b)
y.backward()
# 输出反向传播计算得到的梯度
print(x.grad, w.grad, b.grad)
