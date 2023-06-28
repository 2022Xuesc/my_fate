#!/usr/bin/env python3
import torch.nn

import traceback
from torch.autograd import grad

from federatedml.nn.backend.multi_label.meta_learning.utils import clone_module, update_module


# maml的更新方法
def maml_update(model, lr, grads=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)

    **Description**

    使用梯度和学习率执行一次MAML更新
    该函数对Python对象重新路由，因此避免了原地操作

    注意：模型原地更新，但参数向量不是（原地更新）

    **Arguments**

    * **model** (Module) - The model to update.
    * **lr** (float) - The learning rate used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the gradients in .grad attributes.

    **Example**
    ~~~python
    maml = l2l.algorithms.MAML(Model(), lr=0.1)
    model = maml.clone()
    # 下面两行其实实现了model.adapt(loss)方法，适应新任务
    grads = autograd.grad(loss, model.parameters(), create_graph=True)
    maml_update(model, lr=0.1, grads)
    ~~~
    """
    if grads is not None:
        params = list(model.parameters())
        if not len(grads) == len(list(params)):
            msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
            print(msg)
        # p是参数，g是梯度，为参数计算更新值update
        for p, g in zip(params, grads):
            if g is not None:
                p.update = - lr * g
    else:
        # 根据已经计算好的梯度进行计算
        params = list(model.parameters())
        for p in params:
            if p.grad is not None:
                p.update = - lr * p.grad
    return update_module(model)


# MAML的实现
class MAML(torch.nn.Module):
    """
    **Description**

    High-level implementation of *Model-Agnostic Meta-Learning*.

    This class wraps an arbitrary nn.Module and augments it with `clone()` and `adapt()`
    methods.

    **Example**

    ~~~python
    linear = l2l.algorithms.MAML(nn.Linear(20, 10), lr=0.01)
    clone = linear.clone()
    error = loss(clone(X), y)
    clone.adapt(error)
    error = loss(clone(X), y)
    error.backward()
    ~~~
    """

    def __init__(self, model, lr):
        super(MAML, self).__init__()
        self.module = model
        self.lr = lr

    # 前向传播，调用包装模型的forward方法即可
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    # 在任务上进行适应训练
    def adapt(self, loss, do_calc=True):
        """
        **Description**

        在loss上执行一个梯度步并原地更新克隆后的参数

        **Arguments**

        * **loss** (Tensor) - Loss to minimize upon update.
        """
        gradients = None
        # 计算loss对包装模型参数的梯度
        if do_calc:
            gradients = grad(loss, self.module.parameters())
        # 更新模型
        self.module = maml_update(self.module, self.lr, gradients)

    # 克隆方法
    def clone(self):
        """
        **Description**


        返回模型的一个MAML包装拷贝，其参数和buffer是来自于原模型的torch.clone
        对克隆模型的反向传播损失能够传播到原模型的buffer


        **Arguments**

        * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
            or second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

        """
        # 执行构造函数，返回MAML的包装实例
        return MAML(clone_module(self.module), lr=self.lr)

    def save_classifier_grad(self):
        self.module.fc[0].weight.retain_grad()
        self.module.fc[0].bias.retain_grad()
