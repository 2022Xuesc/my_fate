import torch


# 正确利用相关性计算衍生预测向量
def getCorrectedCandidates(predicts, adjList, neg_adjList, label_prob_vec, requires_grad):
    device = predicts.device
    batch_size = len(predicts)
    _, label_dim = predicts.size()
    candidates = torch.zeros((batch_size, label_dim), dtype=torch.float64).to(device)
    # 遍历每个批次
    for b in range(batch_size):
        # 输入1*C，输出1*C
        predict_vec = predicts[b]
        # 需要推断的标签
        for lj in range(label_dim):
            relation_num = 0
            for li in range(label_dim):
                # 如果li的预测置信度小于0.5，则不能使用该相关性，则应该使用另一种相关性
                # Todo: 两种相关性之间计算一下
                if predict_vec[li] < 0.5:
                    # 如果相等，则直接跳过，因为必然为0
                    if li == lj:
                        continue
                    # Todo: 如果不包含lj这个标签，
                    relation_num += 1
                    if requires_grad:
                        a = neg_adjList[li][lj]
                    else:
                        a = neg_adjList[li][lj].item()
                    # 如果是促进作用
                    if a >= label_prob_vec[lj]:
                        candidates[b][lj] += (1 - predict_vec[li]) * a
                    else:
                        candidates[b][lj] += 1 - (1 - predict_vec[li]) * (1 - a)
                else:
                    relation_num += 1
                    if li == lj:  # 是同一个标签，则1转移
                        candidates[b][lj] += predict_vec[li]
                    else:  # 不是同一个标签，使用相关性值转移
                        if requires_grad:
                            a = adjList[li][lj]
                        else:
                            a = adjList[li][lj].item()
                        # 这里还要对a进行分类讨论
                        # 如果a大于0.5，则促进作用，否则，起抑制作用
                        # 促进作用时，可以直接相乘
                        if a >= label_prob_vec[lj]:
                            candidates[b][lj] += predict_vec[li] * a
                        else:
                            # 如果li值固定，则a越小，lj的概率越小
                            # 如果a值固定，则li概率越大，lj的概率越小
                            # 符合直觉
                            candidates[b][lj] += 1 - predict_vec[li] * (1 - a)
            # 按照转移的标签数量进行平均，促进与抑制综合判定
            # Todo: 这里relation_num可能是0导致无法转移，因此，引入另外一种矩阵
            candidates[b][lj] /= relation_num
    return candidates


# 给定该批次的样本的预测向量，从每个预测向量根据标签相关性重构出其他的标签向量
def getCorrectedCandidates(predicts, adjList, label_prob_vec, requires_grad, predict_gap=0, relation_gap=0):
    device = predicts.device
    batch_size = len(predicts)
    _, label_dim = predicts.size()
    candidates = torch.zeros((batch_size, label_dim), dtype=torch.float64).to(device)

    exists_lower_bound = 0.5 + predict_gap

    # 遍历每个批次
    for b in range(batch_size):
        # 输入1*C，输出1*C
        predict_vec = predicts[b]
        # 需要推断的标签
        for lj in range(label_dim):
            relation_num = 0

            incr_lower_bound = label_prob_vec[lj] + relation_gap
            decr_upper_bound = label_prob_vec[lj] - relation_gap
            for li in range(label_dim):
                if li == lj:  # 是同一个标签，则1转移
                    relation_num += 1
                    candidates[b][lj] += predict_vec[li]
                    continue
                # Todo: 两种相关性之间计算一下
                if predict_vec[li] > exists_lower_bound:  # Todo: 不引入阈值了，这样既考虑促进情况，也考虑抑制的情况
                    if requires_grad:
                        a = adjList[li][lj]
                    else:
                        a = adjList[li][lj].item()
                    relation_num += 1
                    # 标签li对标签lj起促进作用，直接相乘即可
                    if a > incr_lower_bound:
                        # 仅当起促进作用时，才累加
                        candidates[b][lj] += predict_vec[li] * a
                    elif a < decr_upper_bound:  # 标签li对lj起抑制作用
                        candidates[b][lj] += 1 - predict_vec[li] * (1 - a)
            # 按照转移的标签数量进行平均
            # 里边既有促进作用的部分，也有抑制作用的部分
            candidates[b][lj] /= relation_num
    return candidates
