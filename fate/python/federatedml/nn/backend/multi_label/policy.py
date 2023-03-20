import torch
import torch.optim.lr_scheduler
from collections import namedtuple

__all__ = ['LRPolicy', 'ScheduledTrainingPolicy']

PolicyLoss = namedtuple('PolicyLoss', ['overall_loss', 'loss_components'])
LossComponent = namedtuple('LossComponent', ['name', 'value'])


class ScheduledTrainingPolicy(object):

    def __init__(self, classes=None, layers=None):
        self.classes = classes
        self.layers = layers

    def on_epoch_end(self, model, zeros_mask_dict, meta, **kwargs):
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
