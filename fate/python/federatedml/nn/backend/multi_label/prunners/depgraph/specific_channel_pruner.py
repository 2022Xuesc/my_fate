import torch
import torch.nn as nn

import json
import typing
import federatedml.nn.backend.multi_label.prunners.depgraph.ops as ops
import federatedml.nn.backend.multi_label.prunners.depgraph.function as function
import federatedml.nn.backend.multi_label.prunners.depgraph.dependency as dependency


def linear_scheduler(ch_sparsity_dict, steps):
    return [((i) / float(steps)) * ch_sparsity_dict for i in range(steps + 1)]


class SpecificChannelPruner:
    def __init__(
            self,
            model: nn.Module,
            example_inputs: torch.Tensor,
            layer_prune_idxs=None,
            root_module_types=None
    ):
        if root_module_types is None:
            root_module_types = [ops.TORCH_CONV, ops.TORCH_LINEAR]
        self.model = model
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs
        )
        self.root_module_types = root_module_types
        self.layer_prune_idxs = layer_prune_idxs

    # 执行剪枝
    def step(self):
        name2masks = {}
        for name, p in self.model.named_parameters():
            name2masks[name] = torch.ones_like(p)
        for group in self.prune_specific():
            group.prune(name2masks)
        return name2masks

    def prune_specific(self):
        # 这里确定卷积层的名称和需要剪枝的通道
        # Todo: 从文件中读取需要移除的通道数
        # 遍历每个剪枝组，判断组中是否包含目标层的剪枝需要
        for group in self.DG.get_all_groups(root_module_types=self.root_module_types):
            # Todo: 看一下这里group的表示方式
            # 遍历组内的每个依赖，找出目标剪枝组
            for i in range(len(group.items)):
                dep = group.items[i].dep
                name = dep.target._name
                module = dep.target.module
                pruning_fn = dep.handler
                if name in self.layer_prune_idxs and self.DG.is_out_channel_pruning_fn(pruning_fn):
                    # 将当前依赖记录下来？
                    prune_idxs = self.layer_prune_idxs[name]
                    # 获取指定剪枝索引的剪枝组
                    prune_group = self.DG.get_pruning_group(module, pruning_fn, prune_idxs)
                    # 返回剪枝组
                    yield prune_group