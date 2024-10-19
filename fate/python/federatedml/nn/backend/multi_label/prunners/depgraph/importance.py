import abc
import torch
import torch.nn as nn

import typing
import federatedml.nn.backend.multi_label.prunners.depgraph.function as function
import math


# Todo: 学习这里的重要性度量
class Importance(abc.ABC):
    # 可调用的类：给定组group，返回对该组的重要性度量
    @abc.abstractclassmethod
    def __call__(self, group) -> torch.Tensor:
        raise NotImplementedError


# Todo: 基于绝对值的重要性度量
class MagnitudeImportance(Importance):
    def __init__(self, p=2, group_reduction="mean", normalizer="mean"):
        # 使用p范数，默认使用2范数
        self.p = p
        # Todo: 在各个通道上进行平均？和之前的猜测不符
        self.group_reduction = group_reduction
        # 归一化的方法
        self.normalizer = normalizer

    # Todo: 对计算好的组重要性进行归一化
    def _normalize(self, group_importance, normalizer):
        if normalizer is None:
            return group_importance
        # normalizer可调用
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        # 放缩到0-1之间
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (
                    group_importance.max() - group_importance.min() + 1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == "gaussian":
            return (group_importance - group_importance.mean()) / (group_importance.std() + 1e-8)
        else:
            raise NotImplementedError

    # 从组中选出代表性元素，即对其reudce
    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            # Todo: 这里最后为何加上0索引
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction == 'first':
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else:
            raise NotImplementedError
        return group_imp

    # 计算group_importance
    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
        # 遍历组依赖和剪枝索引idxs，应该是所有通道
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            # 对输出通道进行剪枝
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels
            ]:
                # 如果该层经过了转置
                if hasattr(layer, "transposed") and layer.transposed:
                    # Todo: 为什么是在第1维展开
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)  # flatten(1)表示从第1个维度开始展开，保留通道维
                else:
                    w = layer.weight.data[idxs].flatten(1) # 索引需要剪枝的通道
                # 计算层的范数
                local_norm = w.abs().pow(self.p).sum(1)
                # Todo: 当ch_groups > 1时的处理方式
                # if ch_groups > 1:
                #     local_norm = local_norm.view(ch_groups,-1).sum(0)
                #     local_norm = local_norm.repeat(ch_groups)
                group_imp.append(local_norm)
            # 对输入通道进行剪枝
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.flatten(1)
                else:
                    # 对输入通道进行剪枝，因此，需要进行转置;
                    # 卷积层默认的维度是(out_ch, in_ch, k, k)
                    # 全连接层默认的维度是(out_feat, in_feat)
                    w = layer.weight.transpose(0, 1).flatten(1)
                # if ch_groups > 1 and prune_fn == function.prune_conv_in_channels and layer.groups == 1:
                #     # 未分组的卷积层和已分组的卷积层
                #     w = w.view(w.shape[0] // group_imp[0].shape[0],
                #                group_imp[0].shape[0], w.shape[1]).transpose(0, 1).flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                # if ch_groups > 1:
                #     if len(local_norm) == len(group_imp[0]):
                #         local_norm = local_norm.view(ch_groups, -1).sum(0)
                #     local_norm = local_norm.repeat(ch_groups)
                local_norm = local_norm[idxs]
                group_imp.append(local_norm)
            # 对BN层进行剪枝
            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_norm = w.abs().pow(self.p)
                    # if ch_groups  > 1:
                    #     local_norm = local_norm.view(ch_groups,-1).sum(0)
                    #     local_norm = local_norm.repeat(ch_groups)
                    group_imp.append(local_norm)
        if len(group_imp) == 0:
            return None
        imp_size = len(group_imp[0])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp) == imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0) # 最终维度是(valid_layers, channels)，第0维是组涉及到的剪枝层的数目
        group_imp = self._reduce(group_imp) # 对这些不同的层进行reduce，得到组内通道的重要性评价
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp

# LAMP那篇论文重要性度量
class LAMPImportance(MagnitudeImportance):
    pass
