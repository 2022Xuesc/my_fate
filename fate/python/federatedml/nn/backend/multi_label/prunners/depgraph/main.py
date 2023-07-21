import torch
from torchvision.models import resnet101
from dependency import *
from function import *
import importance
from meta_pruner import *


# Todo: 测试依赖图的构建和分组
# model = resnet18(pretrained=True).eval()
# example_inputs = torch.randn(1, 3, 224, 224)
#
# # 构建模型的依赖图
# DG = DependencyGraph().build_dependency(model, example_inputs)
#
# pruning_idxs = [2, 6, 9]
#
# # 对conv1进行剪枝
# pruning_group = DG.get_pruning_group(model.conv1,prune_conv_out_channels,idxs=pruning_idxs)
#
# if DG.check_pruning_group(pruning_group):
#     pruning_group.prune()
#
# print("After pruning:")
# print(model)
#
# print(pruning_group)
#
# # 获取所有的分组
# all_groups = list(DG.get_all_groups())
# print("Number of Groups: %d" % len(all_groups))
# print("The last group:",all_groups[-1])

# Todo: 测试重要性度量
def learn_imp():
    model = resnet101()
    example_inputs = torch.randn(1, 3, 224, 224)
    DG = DependencyGraph().build_dependency(model, example_inputs)
    # 获取conv层的可剪枝索引-->输出层剪枝
    pruning_idxs = list(range(DG.get_out_channels(model.conv1)))
    pruning_group = DG.get_pruning_group(model.conv1, prune_conv_out_channels, pruning_idxs)

    # 度量L1范数
    magnitude_importance = importance.MagnitudeImportance(p=1)
    mag_imp = magnitude_importance(pruning_group)
    print(mag_imp)


def learn_pruner(global_pruning):
    model = resnet101()
    example_inputs = torch.randn(1, 3, 224, 224)
    imp = importance.MagnitudeImportance(p=2)

    # [0, 0, 0.07, 0.14, 0.2, 0.26, 0.33, 0.39, 0.46, 0.55, 0.66]

    ch_sparsities = [0.01 + i * 0.01 for i in range(100)]
    print(ch_sparsities)
    for i in range(len(ch_sparsities)):
        pruner = MetaPruner(
            model,
            example_inputs,
            global_pruning=global_pruning,
            importance=imp,
            ch_sparsity=ch_sparsities[i]
        )
        masks = pruner.step()
        masks = list(masks.values())
        layer_ratios = []
        for j in range(len(masks)):
            layer_mask = masks[j]
            layer_ratios.append((layer_mask.sum() * 1.0 / layer_mask.numel()).item())
        print(layer_ratios)
    print('Hello World')


if __name__ == '__main__':
    # 输出就是组内64个通道的重要性度量
    # learn_imp()
    # 下一步学习实际的剪枝操作
    learn_pruner(global_pruning=True)
    # 如何与fate相结合？
    # Todo:
    #  1. 分组、依赖保留
    #  2. 会生成索引
    #  3. 根据生成的索引确定mask即可
    pass
