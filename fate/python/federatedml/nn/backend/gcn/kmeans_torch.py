import numpy as np
import torch


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        initial_state=None,
        scene_cnts=None
):
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    device = X.device
    initial_state = initial_state.to(device)

    iteration = 0
    # while True:

    dis = pairwise_distance_function(X, initial_state)

    choice_cluster = torch.argmin(dis, dim=1)

    # initial_state_pre = initial_state.clone()

    for index in range(num_clusters):
        selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

        selected = torch.index_select(X, 0, selected)
        # 没有选择的聚类，则直接跳过
        if len(selected) == 0:
            continue
        # 使用加权更新
        w1 = scene_cnts[index]
        weight = w1 + len(selected)
        # Todo: 和原来的场景数加权平均
        initial_state[index] = (w1 * initial_state[index] + selected.sum(dim=0)) / weight

        # center_shift = torch.sum(
        #     torch.sqrt(
        #         torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
        #     ))

        # # increment iteration
        # iteration = iteration + 1
        # 
        # if center_shift ** 2 < tol:
        #     break
    # 返回该批次样本对应的聚类类别和更新后的聚类中心
    # choice_cluster = [cluster_id.item() for cluster_id in choice_cluster]
    return choice_cluster.to(device), initial_state




def pairwise_distance(data1, data2):

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis
