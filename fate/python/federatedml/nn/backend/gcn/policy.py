import torch
import torch.optim.lr_scheduler
from collections import namedtuple

__all__ = ['LRPolicy', 'ScheduledTrainingPolicy', 'RegularizationPolicy']

PolicyLoss = namedtuple('PolicyLoss', ['overall_loss', 'loss_components'])
LossComponent = namedtuple('LossComponent', ['name', 'value'])


class ScheduledTrainingPolicy(object):

    def __init__(self, classes=None, layers=None):
        self.classes = classes
        self.layers = layers

    def on_epoch_end(self, model, meta, **kwargs):
        pass

    # 在反向传播之前，执行一些操作，如添加正则化损失等等
    def before_backward_pass(self, model, epoch, loss, optimizer=None):
        pass


class LRPolicy(ScheduledTrainingPolicy):
    def __init__(self, lr_scheduler):
        super(LRPolicy, self).__init__()
        self.lr_scheduler = lr_scheduler

    def on_epoch_end(self, model, meta, **kwargs):
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(kwargs['metrics'][self.lr_scheduler.mode],
                                   epoch=meta['current_epoch'] + 1)
        else:
            self.lr_scheduler.step(epoch=meta['current_epoch'] + 1)


# 添加正则化的策略
class RegularizationPolicy(ScheduledTrainingPolicy):
    def __init__(self, regularizer):
        super(RegularizationPolicy, self).__init__()
        self.regularizer = regularizer

    def before_backward_pass(self, model, epoch, loss, optimizer=None):
        # 定义正则化损失的类型及寄存设备
        regularizer_loss = torch.tensor(0, dtype=torch.float, device=loss.device)
        # 对每个模型，计算正则化损失
        for param_name, param in model.named_parameters():
            self.regularizer.loss(param, regularizer_loss)
        # 生成PolicyLoss，便于记录分析
        policy_loss = PolicyLoss(loss + regularizer_loss,
                                 [LossComponent(self.regularizer.__class__.__name__ + '_loss', regularizer_loss)])
        return policy_loss
