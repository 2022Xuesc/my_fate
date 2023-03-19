import torch
import torch.optim.lr_scheduler
from collections import namedtuple

__all__ = ['PruningPolicy', 'LRPolicy', 'ScheduledTrainingPolicy', 'QuantizationPolicy', 'RegularizationPolicy']

PolicyLoss = namedtuple('PolicyLoss', ['overall_loss', 'loss_components'])
LossComponent = namedtuple('LossComponent', ['name', 'value'])


class ScheduledTrainingPolicy(object):

    def __init__(self, classes=None, layers=None):
        self.classes = classes
        self.layers = layers

    def on_epoch_begin(self, model, zeros_mask_dict, meta, **kwargs):
        pass

    def on_minibatch_begin(self, model, epoch, minibatch_id, minibatches_per_epoch,
                           zeros_mask_dict, meta, optimizer=None):
        pass

    def before_backward_pass(self, model, epoch, minibatch_id, minibatches_per_epoch, loss, zeros_mask_dict,
                             optimizer=None):
        pass

    def before_parameter_optimization(self, model, epoch, minibatch_id, minibatches_per_epoch,
                                      zeros_mask_dict, meta, optimizer):
        pass

    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer):
        pass

    def on_epoch_end(self, model, zeros_mask_dict, meta, **kwargs):
        pass

    def on_aggregate_end(self):
        pass


class PruningPolicy(ScheduledTrainingPolicy):
    def __init__(self, pruner, pruner_args, classes=None, layers=None):
        super(PruningPolicy, self).__init__(classes, layers)
        self.pruner = pruner
        if pruner_args is None:
            pruner_args = {}
        self.levels = pruner_args.get('levels', None)
        self.keep_mask = pruner_args.get('keep_mask', False)
        self.mini_batch_pruning_frequency = pruner_args.get('mini_batch_pruning_frequency', 0)
        self.mask_on_forward_only = pruner_args.get('mask_on_forward_only', False)  # Todo: 这里是Dynamic Surgery的切入点
        self.mask_gradients = pruner_args.get('mask_gradients', False)
        self.backward_hook_handle = None
        self.use_double_copies = pruner_args.get('use_double_copies', False)
        self.discard_masks_at_minibatch_end = pruner_args.get('discard_masks_at_minibatch_end', False)
        # Initialize state
        self.is_last_epoch = False
        self.is_initialized = False

    def on_epoch_begin(self, model, zeros_mask_dict, meta, **kwargs):
        self.is_last_epoch = meta['current_epoch'] == (meta['ending_epoch'] - 1)
        if self.levels is not None:
            self.pruner.levels = self.levels

        meta['model'] = model
        is_initialized = self.is_initialized

        for param_name, param in model.named_parameters():
            if not is_initialized:
                # Initialize the maskers
                masker = zeros_mask_dict[param_name]
                masker.use_double_copies = self.use_double_copies
                masker.mask_on_forward_only = self.mask_on_forward_only
                # register for the backward hook of the parameters
                if self.mask_gradients:  # 仅在random_filter_pruning中配置为True
                    masker.backward_hook_handle = param.register_hook(masker.mask_gradient)

                self.is_initialized = True
                # 设置掩膜矩阵
                self.pruner.set_param_mask(param, param_name, zeros_mask_dict, meta)
            else:
                self.pruner.set_param_mask(param, param_name, zeros_mask_dict, meta)

    def on_minibatch_begin(self, model, epoch, minibatch_id, minibatches_per_epoch,
                           zeros_mask_dict, meta, optimizer=None):
        set_masks = False
        global_mini_batch_id = epoch * minibatches_per_epoch + minibatch_id
        if ((minibatch_id > 0) and (self.mini_batch_pruning_frequency != 0) and
                (global_mini_batch_id % self.mini_batch_pruning_frequency == 0)):
            # This is _not_ the first mini-batch of a new epoch (performed in on_epoch_begin)
            # and a pruning step is scheduled
            set_masks = True

        for param_name, param in model.named_parameters():
            if set_masks:
                self.pruner.set_param_mask(param, param_name, zeros_mask_dict, meta)
            zeros_mask_dict[param_name].apply_mask(param)

    def before_parameter_optimization(self, model, epoch, minibatch_id, minibatches_per_epoch,
                                      zeros_mask_dict, meta, optimizer):
        for param_name, param in model.named_parameters():
            zeros_mask_dict[param_name].revert_weights(param)

    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer):
        if self.discard_masks_at_minibatch_end:
            for param_name, param in model.named_parameters():
                zeros_mask_dict[param_name].mask = None

    def on_epoch_end(self, model, zeros_mask_dict, meta, **kwargs):
        """The current epoch has ended"""
        if self.is_last_epoch:
            for param_name, param in model.named_parameters():
                masker = zeros_mask_dict[param_name]
                if self.keep_mask:
                    masker.use_double_copies = False
                    masker.mask_on_forward_only = False
                    masker.mask_tensor(param)
                if masker.backward_hook_handle is not None:
                    masker.backward_hook_handle.remove()
                    masker.backward_hook_handle = None


class LRPolicy(ScheduledTrainingPolicy):
    def __init__(self, lr_scheduler):
        super(LRPolicy, self).__init__()
        self.lr_scheduler = lr_scheduler

    def on_epoch_end(self, model, zeros_mask_dict, meta, **kwargs):
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(kwargs['metrics'][self.lr_scheduler.mode],
                                   epoch=meta['current_epoch'] + 1)
        else:
            self.lr_scheduler.step(epoch=meta['current_epoch'] + 1)


class QuantizationPolicy(ScheduledTrainingPolicy):
    def __init__(self, quantizer):
        super(QuantizationPolicy, self).__init__()
        self.quantizer = quantizer
        # 准备模型+量化参数
        self.quantizer.prepare_model()

    def on_minibatch_begin(self, model, epoch, minibatch_id, minibatches_per_epoch,
                           zeros_mask_dict, meta, optimizer=None):
        # Todo: 在开始前对其进行量化
        self.quantizer.quantize_params()

    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer):
        # 在模型更新后，对参数再次进行量化
        # Todo: 确保训练完成时(验证阶段)模型参数处于量化状态
        self.quantizer.quantize_params()

    def on_aggregate_end(self):
        # Todo: 聚合后需要对weight继续量化，从而开始继续迭代
        self.quantizer.quantize_params()


class RegularizationPolicy(ScheduledTrainingPolicy):

    def __init__(self, regularizer, keep_mask=False):
        super(RegularizationPolicy, self).__init__()
        self.regularizer = regularizer
        self.keep_mask = keep_mask
        self.is_last_epoch = False

    def on_epoch_begin(self, model, zeros_mask_dict, meta, **kwargs):
        self.is_last_epoch = meta['current_epoch'] == (meta['ending_epoch'] - 1)

    # 在反向传播之前,添加正则项的损失
    def before_backward_pass(self, model, epoch, minibatch_id, minibatches_per_epoch, loss,
                             zeros_mask_dict, optimizer=None):
        regularizer_loss = torch.tensor(0, dtype=torch.float, device=loss.device)

        for param_name, param in model.named_parameters():
            self.regularizer.loss(param, param_name, regularizer_loss, zeros_mask_dict)

        policy_loss = PolicyLoss(loss + regularizer_loss,
                                 [LossComponent(self.regularizer.__class__.__name__ + '_loss', regularizer_loss)])
        return policy_loss

    # Todo: 跳过该正则项中计算掩膜矩阵的步骤
    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer):
        if self.regularizer.threshold_criteria is None:
            return

        keep_mask = False
        if (minibatches_per_epoch - 1 == minibatch_id) and self.is_last_epoch and self.keep_mask:
            # If this is the last mini_batch in the last epoch, and the scheduler wants to
            # keep the regularization mask, then now is the time ;-)
            keep_mask = True

        for param_name, param in model.named_parameters():
            self.regularizer.threshold(param, param_name, zeros_mask_dict)
            if keep_mask:
                zeros_mask_dict[param_name].is_regularization_mask = False
            zeros_mask_dict[param_name].apply_mask(param)
