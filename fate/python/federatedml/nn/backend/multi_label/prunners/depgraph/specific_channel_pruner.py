import torch
import torch.nn as nn
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

    # 执行剪枝
    def step(self):
        name2masks = {}
        for name, p in self.model.named_parameters():
            name2masks[name] = torch.ones_like(p)
        for group in self.prune_specific():
            group.prune(name2masks)
        return name2masks

    def prune_specific(self):
        layer_prune_idxs = {
            "conv1": [4, 3, 32, 20, 60, 11],
            "layer1.2.conv3": [36, 237, 94, 162, 3, 136, 127, 163, 201, 56, 112, 129, 209, 86, 60, 181, 49, 95, 223, 41,
                               35, 194, 168, 78, 146],
            "layer2.3.conv3": [8, 23, 32, 48, 57, 62, 86, 94, 126, 128, 140, 153, 199, 201, 206, 225, 239, 256, 272,
                               286, 316, 343, 347, 353, 362, 366, 368, 375, 386, 410, 416, 422, 446, 457, 462, 463, 467,
                               472, 506, 508, 510, 98, 142, 372, 492, 92, 61, 214, 115, 150, 503],
            "layer3.22.conv3": [1, 21, 22, 26, 30, 38, 40, 50, 58, 60, 72, 99, 104, 110, 114, 133, 157, 161, 175, 207,
                                212, 218, 226, 235, 239, 260, 266, 280, 299, 314, 324, 326, 346, 351, 380, 388, 390,
                                392, 404, 411, 425, 426, 449, 466, 470, 480, 493, 494, 509, 510, 512, 514, 515, 531,
                                550, 558, 570, 577, 588, 593, 596, 597, 603, 608, 624, 625, 639, 649, 658, 661, 669,
                                673, 697, 702, 705, 712, 725, 730, 739, 773, 774, 793, 810, 844, 845, 859, 887, 888,
                                891, 893, 904, 914, 932, 935, 943, 958, 976, 979, 982, 987, 994, 1018],
            "layer4.2.conv3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                               25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                               47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
                               69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                               91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                               110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
                               128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
                               146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
                               164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
                               182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
                               200, 228, 1200, 1680                               ]
        }
        # 遍历每个剪枝组，判断组中是否包含目标层的剪枝需要
        for group in self.DG.get_all_groups(root_module_types=self.root_module_types):
            # Todo: 看一下这里group的表示方式
            # 遍历组内的每个依赖，找出目标剪枝组
            for i in range(len(group.items)):
                dep = group.items[i].dep
                name = dep.target._name
                module = dep.target.module
                pruning_fn = dep.handler
                if name in layer_prune_idxs and self.DG.is_out_channel_pruning_fn(pruning_fn):
                    # 将当前依赖记录下来？
                    prune_idxs = layer_prune_idxs[name]
                    # 获取指定剪枝索引的剪枝组
                    prune_group = self.DG.get_pruning_group(module, pruning_fn, prune_idxs)
                    # 返回剪枝组
                    yield prune_group
