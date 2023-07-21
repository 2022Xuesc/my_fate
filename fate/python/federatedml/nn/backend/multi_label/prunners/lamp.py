import torch
from torch.nn.utils import prune
from utils import get_weights, get_modules
import numpy as np


# 使用lamp方法对权重进行剪枝
def prune_weights_lamp(model, amount):
    # 计算每一层的稀疏水平
    amounts = compute_lamp_amounts(model, amount)
    # 使用torch的剪枝方法进行剪枝
    prune_weights_l1predefined(model, amounts)


def prune_weights_l1predefined(model, amounts):
    # 获取所有可剪枝模块
    # Todo: 可剪枝模块参数和所有优化参数的隶属情况
    mlist = get_modules(model)
    # 对于每个模块，使用torch自带的方法进行剪枝
    for idx, m in enumerate(mlist):
        prune.l1_unstructured(m, name='weight', amount=float(amounts[idx]))


# 给定模型和全局稀疏度amount，返回每一层的剪枝率
def compute_lamp_amounts(model, amount):
    # 获取每一层未剪枝权重的数量
    unmasked = count_unmasked_weights(model)
    num_survive = int(np.round(unmasked.sum() * (1.0 - amount)))

    # 计算模型每层每个位置的lamp分数
    flattened_scores = [normalize_scores(w ** 2).view(-1) for w in get_weights(model)]

    concat_scores = torch.cat(flattened_scores, dim=0)
    topks, _ = torch.topk(concat_scores, num_survive)
    threshold = topks[-1]  # 计算出阈值

    final_survs = [torch.ge(score, threshold * torch.ones(score.size()).to(score.device)).sum() for score in
                   flattened_scores]
    amounts = []
    for idx, final_surv in enumerate(final_survs):
        amounts.append(1.0 - (final_surv / unmasked[idx]))

    return amounts


def normalize_scores(scores):
    """
    LAMP的归一化方案
    """
    # 已递增顺序排列分数
    # .view(-1)展平，然后排序
    sorted_scores, sorted_idx = scores.view(-1).sort(descending=False)
    # 计算每个位置的权重的累积和
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)  # 计算前缀和
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape, device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp) - 1]
    # 使用cum sum进行归一化
    sorted_scores /= (scores.sum() - scores_cumsum)
    # 整理并输出
    new_scores = torch.zeros(scores_cumsum.shape, device=scores.device)
    new_scores[sorted_idx] = sorted_scores

    return new_scores.view(scores.shape)


def count_unmasked_weights(model):
    """
    返回未剪枝权重的1维张量，每个模块对应一个分量
    """
    mlist = get_modules(model)
    unmasked = []
    # 对于每个可剪枝模块，统计其剩余的未剪枝模块的数量
    for m in mlist:
        unmasked.append(m.weight_mask.sum())
    return torch.FloatTensor(unmasked)
