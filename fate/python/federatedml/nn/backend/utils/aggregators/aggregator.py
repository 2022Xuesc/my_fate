import numpy as np


# 定义若干聚合方法

def aggregate_bn_data(bn_tensors, degrees=None):
    
    degrees = np.array(degrees)
    degrees_sum = degrees.sum(axis=0)
    total_weight = degrees_sum if degrees.ndim == 1 else degrees_sum[-1]
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
            client_weight = degrees[idx] if degrees.ndim == 1 else degrees[idx][-1]
            mean += client_mean * client_weight
        mean /= total_weight
        bn_data.append(mean)
        # 计算完均值之后，开始计算方差
        var_idx = mean_idx + 1
        var = np.zeros(mean_var_dim)
        for idx in range(client_nums):
            client_mean = bn_tensors[idx][mean_idx]
            client_var = bn_tensors[idx][var_idx]
            client_weight = degrees[idx] if degrees.ndim == 1 else degrees[idx][-1]
            var += (client_var + client_mean ** 2 - mean ** 2) * client_weight
        var /= total_weight
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
    total_weight = degrees_sum if degrees.ndim == 1 else degrees_sum[-1]
    for i in range(len(tensors)):
        client_weight = degrees[i] if degrees.ndim == 1 else degrees[i][-1]
        for j, tensor in enumerate(tensors[i]):
            tensor *= client_weight
            tensor /= total_weight
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


def aggregate_scene_adjs_with_cnts(scene_infos):
    num_clients = len(scene_infos)
    num_scenes = len(scene_infos[0][0])
    num_labels = len(scene_infos[0][1][0])
    fixed_adjs = np.zeros((num_clients, num_scenes, num_labels, num_labels))
    names = [scene_infos[i][3] for i in range(num_clients)]
    for i in range(num_clients):
        linear_i = scene_infos[i][0]
        for k, scene_k in enumerate(linear_i):
            # 记录每个权重和对应的场景
            coefficients = [None] * num_clients
            total_cnt = scene_infos[i][2][k]  # 当前客户端第k个场景下的图像数
            cosine_similarities = np.zeros(num_scenes)
            # 遍历每个其他客户端j
            for j in range(num_clients):
                if j == i:
                    continue
                linear_j = scene_infos[j][0]
                for l, scene_l in enumerate(linear_j):
                    dot_product = np.dot(scene_k, scene_l)
                    norm_vector1 = np.linalg.norm(scene_k)
                    norm_vector2 = np.linalg.norm(scene_l)
                    cosine_similarities[l] = dot_product / (norm_vector1 * norm_vector2)
                # 从余弦相似度选出最大的那个场景，记录场景id和场景相似度
                max_scene_id = cosine_similarities.argmax()  # Todo: 如果余弦相似度为负数，则不进行聚合
                max_scene_similarity = cosine_similarities[max_scene_id]
                # 如果相似度为负数，则跳过当前客户端
                if max_scene_similarity < 0:
                    continue
                coefficients[j] = [max_scene_id, max_scene_similarity]
                total_cnt += scene_infos[j][2][max_scene_id]
            # 根据场景贡献对权重进行修正
            # 再次遍历每个客户端
            other_weights = 0
            agg_scene_adj = np.zeros((num_labels, num_labels))
            for j in range(num_clients):
                if coefficients[j] is not None:
                    # 这里的weight是余弦相似度，还要对其进行场景数的修正
                    scene_id, weight = coefficients[j]
                    coefficients[j][1] = weight * scene_infos[j][2][scene_id] / total_cnt
                    other_weights += coefficients[j][1]
                    # Todo: 这里即应该是修正后的权重coefficients[j][1]
                    agg_scene_adj += scene_infos[j][1][scene_id] * coefficients[j][1]
            self_coefficient = 1 - other_weights
            agg_scene_adj += self_coefficient * scene_infos[i][1][k]
            # 进行聚合
            fixed_adjs[i][k] = agg_scene_adj
    return (names, fixed_adjs)


# infos[0]为scene_linear参数
# infos[1]为scene_adjs
# infos[2]为scene_cnts
# 返回的应该是每个客户端每个场景的邻接矩阵，即scene_adjs
# Todo: 这里不应该考虑scene_cnts
def aggregate_scene_adjs(scene_infos):
    num_clients = len(scene_infos)
    num_scenes = len(scene_infos[0][0])
    num_labels = len(scene_infos[0][1][0])
    fixed_adjs = np.zeros((num_clients, num_scenes, num_labels, num_labels))
    names = [scene_infos[i][3] for i in range(num_clients)]
    for i in range(num_clients):
        linear_i = scene_infos[i][0]
        for k, scene_k in enumerate(linear_i):
            # 记录每个权重和对应的场景
            coefficients = [None] * num_clients
            cosine_similarities = np.zeros(num_scenes)
            # 遍历每个其他客户端j
            for j in range(num_clients):
                if j == i:
                    continue
                linear_j = scene_infos[j][0]
                for l, scene_l in enumerate(linear_j):
                    dot_product = np.dot(scene_k, scene_l)
                    norm_vector1 = np.linalg.norm(scene_k)
                    norm_vector2 = np.linalg.norm(scene_l)
                    cosine_similarities[l] = dot_product / (norm_vector1 * norm_vector2)
                # 从余弦相似度选出最大的那个场景，记录场景id和场景相似度
                max_scene_id = cosine_similarities.argmax()  # Todo: 如果余弦相似度为负数，则不进行聚合
                max_scene_similarity = cosine_similarities[max_scene_id]
                # 如果相似度为负数，则跳过当前客户端
                if max_scene_similarity < 0:
                    continue
                coefficients[j] = [max_scene_id, max_scene_similarity]
            # 再次遍历每个客户端
            agg_scene_adj = np.zeros((num_labels, num_labels))
            # 算上自身的相似度，即聚合权重为1
            total_weight = 1
            for j in range(num_clients):
                if coefficients[j] is not None:
                    # 这里的weight是余弦相似度，还要对其进行场景数的修正
                    scene_id, weight = coefficients[j]
                    total_weight += coefficients[j][1]
                    agg_scene_adj += scene_infos[j][1][scene_id] * coefficients[j][1]
            agg_scene_adj += scene_infos[i][1][k]
            agg_scene_adj /= total_weight
            # 进行聚合
            fixed_adjs[i][k] = agg_scene_adj
    return names, fixed_adjs
