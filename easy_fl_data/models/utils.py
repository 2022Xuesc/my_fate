import numpy as np
import torch


def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    # 这里result['adj']是80*80维的矩阵，表示标签之间的共生矩阵
    # 而result['nums']表示各个标签出现的次数
    _adj = result['adj']
    _nums = result['nums']
    # 长度为80的一维向量，扩充一个新的维度，成80*1维的矩阵
    _nums = _nums[:, np.newaxis]
    # 转换成概率的形式，注意这里同时除以_nums之后不对称了,第i行的除以标签i的出现次数
    _adj = _adj / _nums
    # 为了训练的鲁棒性，小于阈值t则设置为0，否则设置为1
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    # 这里对第0维求和，保持第1维的形状，得到1*80的矩阵；第i列的除以对应的和，可认为是归一化的一个过程
    # Todo: 但是这个公式为什么这样设计
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj


# Todo: 给定图卷积神经网络的相关性矩阵A，对A进行特殊处理
def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    # 这一步执行的运算其实是D^T * A^T * D
    # 前面已经经过转置了，那这一步应该不用对A进行转置
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


# 为每个场景都生成一个邻接矩阵
def gen_adjs(A):
    batch_size = A.size(0)
    adjs = torch.zeros_like(A)
    for i in range(batch_size):
        # 这里对行求和
        D = torch.pow(A[i].sum(1).float(), -0.5)
        # 将其转换成对角矩阵
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A[i], D).t(), D)
        adjs[i] = adj
    return adjs