import numpy as np


# 定义若干聚合方法

def aggregate_bn_data(bn_tensors, degrees):
    degrees = np.array(degrees)
    degrees_sum = degrees.sum(axis=0)

    client_nums = len(bn_tensors)
    layer_nums = len(bn_tensors[0]) // 2
    bn_data = []
    # 遍历每一层
    for i in range(layer_nums):
        mean_idx = i * 2
        mean_var_dim = len(bn_tensors[0][mean_idx])
        mean = np.zeros(mean_var_dim)
        # 遍历每个客户端
        for idx in range(client_nums):
            # 该层在该客户端上的mean是bn_tensors[id][i * 2],方差是bn_tensors[id][i * 2 + 1]
            client_mean = bn_tensors[idx][mean_idx]
            mean += client_mean * degrees[idx][-1]
        mean /= degrees_sum[-1]
        bn_data.append(mean)
        # 计算完均值之后，开始计算方差
        var_idx = mean_idx + 1
        var = np.zeros(mean_var_dim)
        for idx in range(client_nums):
            client_mean = bn_tensors[idx][mean_idx]
            client_var = bn_tensors[idx][var_idx]
            var += (client_var + client_mean ** 2 - mean ** 2) * degrees[idx][-1]
        var /= degrees_sum[-1]
        bn_data.append(var)
    return bn_data


# 根据标签进行聚合
def aggregate_by_labels(tensors, degrees):
    # degrees是91个元素的列表，前90个元素是最后一层各个类别的聚合权重，而最后一个元素是之前层的聚合权重
    # 先聚合之前的特征层，聚合权重为degrees[i][-1]
    # 将degree转为array
    degrees = np.array(degrees)
    degrees_sum = degrees.sum(axis=0)
    # i表示第i个客户端
    for i in range(len(tensors)):
        for j, tensor in enumerate(tensors[i]):
            # 如果是最后两层
            if j == len(tensors[i]) - 2 or j == len(tensors[i]) - 1:
                # 对每个列向量进行聚合
                for k in range(len(tensor)):
                    # 对col_vec进行聚合
                    # 如果客户端都不含对应标签的数据，则使用传统方法进行聚合，使得聚合后的权重非0
                    if degrees_sum[k] == 0:
                        tensor[k] *= degrees[i][-1]
                        tensor[k] /= degrees_sum[-1]
                    else:
                        tensor[k] *= degrees[i][k]
                        tensor[k] /= degrees_sum[k]
                    if i != 0:
                        tensors[0][j][k] += tensor[k]
            else:
                tensor *= degrees[i][-1]
                tensor /= degrees_sum[-1]
                if i != 0:
                    tensors[0][j] += tensor
    # 聚合后的权重即为tensors[0]
    return tensors[0]


def aggregate_whole_model(tensors, degrees):
    degrees = np.array(degrees)
    degrees_sum = degrees.sum(axis=0)
    for i in range(len(tensors)):
        for j, tensor in enumerate(tensors[i]):
            tensor *= degrees[i][-1]
            tensor /= degrees_sum[-1]
            if i != 0:
                tensors[0][j] += tensor
    return tensors[0]


def aggregate_relation_matrix(relation_matrices, degrees):
    degrees = np.array(degrees)
    degrees_sum = degrees.sum(axis=0)
    client_nums = len(relation_matrices)
    relation_matrix = np.zeros_like(relation_matrices[0])
    for i in range(client_nums):
        relation_matrix += relation_matrices[i] * degrees[i][-1] / degrees_sum[-1]
    return relation_matrix
