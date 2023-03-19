from functools import partial
import numpy as np
import math
import logging
import torch
from random import uniform

__all__ = ["LpRankedStructureParameterPruner",
           "L1RankedStructureParameterPruner",
           "L2RankedStructureParameterPruner"
           ]

from federatedml.nn.backend.distiller import norms, thresholding, utils
import federatedml.nn.backend.distiller.utils as utils

msglogger = logging.getLogger(__name__)


# Todo: 其中是否包含级联删除？   group_dependency？
class _RankedStructureParameterPruner(object):
    """Base class for pruning structures by ranking them.
    """

    def __init__(self, name, group_type, sparsities,
                 group_dependency=None, group_size=1, rounding_fn=math.floor, noise=0.):
        self.name = name
        self.group_type = group_type
        self.group_dependency = group_dependency
        self.sparsities = sparsities
        self.leader_binary_map = None
        self.last_target_sparsity = None
        self.group_size = group_size
        self.rounding_fn = rounding_fn
        self.noise = noise

    def leader(self):
        # "leader"是参数列表中的第一个权重向量
        # Todo: 待修正
        return self.sparsities[0]

    def fraction_to_prune(self, param_name):
        if param_name not in self.sparsities:
            if '*' not in self.sparsities:
                return
            else:
                sparsity = self.sparsities['*']
        else:
            sparsity = self.sparsities[param_name]
        return sparsity

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        fraction_to_prune = self.fraction_to_prune(param_name)
        try:
            model = meta['model']
        except TypeError:
            model = None
        return self._set_param_mask_by_sparsity_target(param, param_name, zeros_mask_dict, fraction_to_prune, model)

    def _set_param_mask_by_sparsity_target(self, param, param_name, zeros_mask_dict, target_sparsity, model):
        # 不需要进行剪枝
        if target_sparsity is None:
            return
        binary_map = None
        if self.group_dependency == "Leader":
            if target_sparsity != self.last_target_sparsity:
                # Todo: 剪枝是在参数层按序进行的，但何时开始新的一次剪枝对于Pruner对象来说是透明的；
                # Todo: 因此，需要对稀疏水平进行记录，如果稀疏水平发生变化，说明是一次全新的剪枝；需要对leader_binary_map进行更新
                self.last_target_sparsity = target_sparsity
                self.leader_binary_map = self.prune_group(target_sparsity, model.state_dict()[self.leader()],
                                                          self.leader(), zeros_mask_dict=None)
            assert self.leader_binary_map is not None
            # Todo: 注意，每次剪枝过程都需要leader层的掩膜结果
            binary_map = self.leader_binary_map
        # 将实际的剪枝操作推迟到子类实现中
        self.prune_group(target_sparsity, param, param_name, zeros_mask_dict, model, binary_map)

    def prune_group(self, fraction_to_prune, param, param_name, zeros_mask_dict, model=None, binary_map=None):
        raise NotImplementedError


class LpRankedStructureParameterPruner(_RankedStructureParameterPruner):
    """使用Lp范数对滤波器进行排序和剪枝

    def __init__(self, name, group_type, sparsities,
                 group_dependency=None, group_size=1, rounding_fn=math.floor, noise=0.):
    """

    def __init__(self, name, group_type, sparsities,
                 group_dependency=None, magnitude_fn=None,
                 noise=0.0, group_size=1, rounding_fn=math.floor):
        super().__init__(name, group_type, sparsities, group_dependency,
                         group_size, rounding_fn, noise)
        # Todo: 注意，这里的3D和Filters等价，Channels和Rows等价
        if group_type not in ['3D', 'Filters', 'Channels', 'Rows', 'Blocks']:
            raise ValueError("Structure {} was requested but "
                             "currently ranking of this shape is not supported".
                             format(group_type))
        assert magnitude_fn is not None
        self.magnitude_fn = magnitude_fn

    def prune_group(self, fraction_to_prune, param, param_name, zeros_mask_dict, model=None, binary_map=None):
        if fraction_to_prune == 0:
            return
        group_pruning_fn = None
        if self.group_type in ('3D', 'Filters'):
            group_pruning_fn = partial(self.rank_and_prune_filters, noise=self.noise)
        elif self.group_type in ('Channels', 'Rows'):
            group_pruning_fn = partial(self.rank_and_prune_channels, noise=self.noise)

        binary_map = group_pruning_fn(fraction_to_prune, param, param_name,
                                      zeros_mask_dict, binary_map,
                                      magnitude_fn=self.magnitude_fn,
                                      group_size=self.group_size)
        return binary_map

    @staticmethod
    # Todo: 对通道进行排序并剪枝
    def rank_and_prune_channels(fraction_to_prune, param, param_name=None, zeros_mask_dict=None,
                                binary_map=None, magnitude_fn=norms.l1_norm,
                                noise=0.0, group_size=1, rounding_fn=math.floor):
        if binary_map is None:
            bottomk_channels, channel_mags = norms.rank_channels(param, group_size, magnitude_fn,
                                                                 fraction_to_prune, rounding_fn, noise)
            if bottomk_channels is None:
                # 空列表意为fraction_to_pruner太小而无法剪枝
                return
            # 返回的数组是升序排列的最小的k个通道范数值，因此，最后一个即为剪枝阈值。
            threshold = bottomk_channels[-1]
            # 将通道范数与剪枝阈值进行比较，得到掩膜矩阵
            binary_map = channel_mags.gt(threshold).type(param.data.type())

        if zeros_mask_dict is not None:
            mask, _ = thresholding.expand_binary_map(param, 'Channels', binary_map)
            zeros_mask_dict[param_name].mask = mask
            msglogger.info("%sRankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                           magnitude_fn, param_name,
                           utils.sparsity_ch(zeros_mask_dict[param_name].mask),
                           fraction_to_prune, binary_map.sum().item(), param.size(1))
        return binary_map

    @staticmethod
    # 对滤波器进行排序并剪枝
    def rank_and_prune_filters(fraction_to_prune, param, param_name, zeros_mask_dict,
                               binary_map=None, magnitude_fn=norms.l1_norm,
                               noise=0.0, group_size=1, rounding_fn=math.floor):
        assert param.dim() == 4 or param.dim() == 3, "This pruning is only supported for 3D and 4D weights"
        if binary_map is None:
            bottomk_filters, filter_mags = norms.rank_filters(param, group_size, magnitude_fn,
                                                              fraction_to_prune, rounding_fn, noise)
            if bottomk_filters is None:
                msglogger.info("Too few filters - can't prune %.1f%% filters", 100 * fraction_to_prune)
                return
            threshold = bottomk_filters[-1]
            binary_map = filter_mags.gt(threshold).type(param.data.type())

        if zeros_mask_dict is not None:
            mask, _ = thresholding.expand_binary_map(param, 'Filters', binary_map)
            zeros_mask_dict[param_name].mask = mask
            msglogger.info("%sRankedStructureParameterPruner - param: %s pruned=%.3f goal=%.3f",
                           magnitude_fn, param_name,
                           utils.sparsity(mask),
                           fraction_to_prune)
        return binary_map


class L1RankedStructureParameterPruner(LpRankedStructureParameterPruner):
    """Uses mean L1-norm to rank and prune structures.

    This class prunes to a prescribed percentage of structured-sparsity (level pruning).
    """

    def __init__(self, name, group_type, sparsities,
                 group_dependency=None, noise=0.0,
                 group_size=1, rounding_fn=math.floor):
        super().__init__(name, group_type, sparsities, group_dependency,
                         magnitude_fn=norms.l1_norm, noise=noise,
                         group_size=group_size, rounding_fn=rounding_fn)


class L2RankedStructureParameterPruner(LpRankedStructureParameterPruner):
    """Uses mean L2-norm to rank and prune structures.

    This class prunes to a prescribed percentage of structured-sparsity (level pruning).
    """

    def __init__(self, name, group_type, sparsities,
                 group_dependency=None, noise=0.0,
                 group_size=1, rounding_fn=math.floor):
        super().__init__(name, group_type, sparsities, group_dependency,
                         magnitude_fn=norms.l2_norm, noise=noise,
                         group_size=group_size, rounding_fn=rounding_fn)


def _mask_from_filter_order(filters_ordered_by_criterion, param, num_filters, binary_map):
    if binary_map is None:
        # Todo: distiller源码这里为什么要.cuda()
        binary_map = torch.zeros(num_filters)
        binary_map[filters_ordered_by_criterion] = 1
    return thresholding.expand_binary_map(param, "Filters", binary_map)


class ActivationRankedFilterPruner(_RankedStructureParameterPruner):
    def __init__(self, name, group_type, sparsities, group_dependency=None):
        super().__init__(name, group_type, sparsities, group_dependency)

    @property
    def activation_rank_criterion(self):
        raise NotImplementedError

    def prune_group(self, fraction_to_prune, param, param_name, zeros_mask_dict, model=None, binary_map=None):
        if fraction_to_prune == 0:
            return
        binary_map = self.rank_and_prune_filters(fraction_to_prune, param, param_name,
                                                 zeros_mask_dict, model, binary_map)
        return binary_map

    def rank_and_prune_filters(self, fraction_to_prune, param, param_name, zeros_mask_dict, model, binary_map=None):
        assert param.dim() == 4, "This pruning is only supported for 4D weights"
        fq_name = param_name.replace(".conv", ".relu")[:-len(".weight")]
        utils.assign_layer_fq_names(model)
        module = utils.find_module_by_fq_name(model, fq_name)
        assert module is not None
        if not hasattr(module, self.activation_rank_criterion):
            raise ValueError("需要指定对激活单元的排序标准")
        quality_criterion, std = getattr(module, self.activation_rank_criterion).value()
        num_filters = param.size(0)
        num_filters_to_prune = int(fraction_to_prune * num_filters)
        if num_filters_to_prune == 0:
            msglogger.info("Too few filters - can't prune %.1f%% filters", 100 * fraction_to_prune)
            return
        # Todo: np.argsort函数的学习
        filters_ordered_by_criterion = np.argsort(quality_criterion)[:-num_filters_to_prune]
        mask, binary_map = _mask_from_filter_order(filters_ordered_by_criterion, param, num_filters, binary_map)
        zeros_mask_dict[param_name] = mask

        return binary_map


class ActivationAPoZRankedFilterPruner(ActivationRankedFilterPruner):
    @property
    def activation_rank_criterion(self):
        return 'apoz_channels'


class ActivationMeanRankedFilterPruner(ActivationRankedFilterPruner):
    @property
    def activation_rank_criterion(self):
        return "mean_channels"
