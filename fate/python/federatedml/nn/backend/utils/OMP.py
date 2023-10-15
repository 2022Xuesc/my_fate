import torch
import numpy as np
from federatedml.nn.backend.utils.relations.candidater import *

# 导出为工具包
# 找到arr中下标不在集合S中的最大值对应的下标
# 并且下标不能等于当前图像i
def findMaxIndex(arr, S, cur):
    maxVal = 0
    maxIndex = -1
    for i in range(len(arr)):
        if i not in S and i != cur:
            if arr[i] > maxVal:
                maxVal = arr[i]
                maxIndex = i
    return maxIndex

    # features是该批次中的特征
    # predicts是该批次的预测值


# predicts：该批次样本b的预测向量，维度是b*C
# adjList：维护与每个标签相关的标签以及对应的值
def LabelOMP(predicts, adjList, corrected=False):
    device = predicts.device
    batch_size = len(predicts)
    _, label_dim = predicts.size()
    # 每张图片，找和它最相似的k张图片
    k = batch_size // 2

    # 最终输出的是图像之间的特征相似度矩阵和语义相似度矩阵
    predict_similarities = torch.zeros(batch_size, batch_size, dtype=torch.float64).to(device)
    # Todo: candidates重复计算了啊
    # candidates表示从一个标签预测向量中根据标签相关性推断出来的新预测向量
    if not corrected:
        candidates = getCandidates(predicts, adjList, requires_grad=False)
    else:
        candidates = getCorrectedCandidates(predicts, adjList, requires_grad=False)
    # 对第1维计算范数
    candidate_norms = torch.norm(candidates, dim=1)

    # 遍历每张图片
    for i in range(batch_size):
        # 需要拟合的残差
        predict = predicts[i]

        # 进行k次迭代
        # Todo: 维护相似集合，以及相似图像的特征向量和预测向量
        S = set()
        indexes = []
        candidateX = torch.empty(0, label_dim, dtype=torch.float64).to(device)

        # 现在可以计算内积了
        candidate_inner_products = torch.matmul(candidates, predict)

        predict_scores = candidate_inner_products / candidate_norms
        for j in range(k):
            # 找到最相似的图像i‘
            # 从中选出相似性最高的图像，加到相似集中
            index = findMaxIndex(predict_scores, S, i)
            # 判断是否满足内积大于等于0的条件
            # 如果不满足，说明找不到相似图片，直接退出即可
            if torch.matmul(predicts[index], predict) <= 0:
                break
            S.add(index)
            indexes.append(index)

            candidateX = torch.cat((candidateX, candidates[index].unsqueeze(0)), dim=0)

            # Todo: 这里不应该预测值拟合，而应该是预测值经相关性的拟合？
            predict_coefficients = torch.linalg.lstsq(torch.transpose(candidateX, 0, 1), predict)[0]
            # 更新相似性矩阵
            # 不是对称的，因此，更新第i行
            for m in range(len(indexes)):
                neighbor = indexes[m]
                # Todo: 验证引入的约束操作对与原解的修改情况
                predict_similarities[i][neighbor] = max(0, predict_coefficients[m])  # 确保相似性大于0
    return predict_similarities.detach()  # 对于无需通过pytorch计算图优化的变量，将其detach


def OMP(features, predicts, A):
    # 获取设备
    device = features.device
    # 先把features展平
    batch_size = len(features)
    features = features.reshape(batch_size, -1)
    feature_dim = features.size()[1]
    _, label_dim = predicts.size()
    # 每张图片，仅找和它最相似的k张图片
    k = batch_size // 2

    # 正则化系数
    lambda_y = 0.5
    lambda_f = 0.5

    # 最终输出的是图像之间的特征相似度矩阵和语义相似度矩阵
    feature_similarities = torch.zeros(batch_size, batch_size, dtype=torch.float64).to(device)
    predict_similarities = torch.zeros(batch_size, batch_size, dtype=torch.float64).to(device)

    # 遍历每张图片
    for i in range(batch_size):
        # 需要拟合的残差
        predict = predicts[i]
        feature = features[i]

        # 进行k次迭代
        # Todo: 维护相似集合，以及相似图像的特征向量和预测向量
        S = set()
        indexes = []
        featureX = torch.empty(0, feature_dim, dtype=torch.float64).to(device)
        predictX = torch.empty(0, label_dim, dtype=torch.float64).to(device)
        for j in range(k):
            # 找到最相似的图像i‘
            predict_inner_products = torch.matmul(torch.matmul(predicts, A), predict)
            # 对第1维计算范数
            predict_norms = torch.norm(predicts, dim=1)
            predict_scores = predict_inner_products / predict_norms

            feature_inner_products = torch.matmul(features, feature)
            feature_norms = torch.norm(features, dim=1)
            feature_scores = feature_inner_products / feature_norms

            score = lambda_y * predict_scores + lambda_f * feature_scores

            # Todo: 从中选出相似性最高的图像，加到相似集中
            index = findMaxIndex(score, S, i)
            # 判断是否满足内积大于等于0的条件
            # 如果不满足，说明找不到相似图片，直接退出即可
            if torch.matmul(predicts[index], predict) <= 0 or torch.matmul(features[index], feature) <= 0:
                break
            S.add(index)
            indexes.append(index)
            # 检查一下features[index]是否有问题

            featureX = torch.cat((featureX, features[index].unsqueeze(0)), dim=0)
            predictX = torch.cat((predictX, predicts[index].unsqueeze(0)), dim=0)

            # Todo: 这里可能会报错，捕捉到异常后
            feature_coefficients = torch.linalg.lstsq(torch.transpose(featureX, 0, 1), feature)[0]
            predict_coefficients = torch.linalg.lstsq(torch.transpose(predictX, 0, 1), predict)[0]
            # 更新相似性矩阵
            # 不是对称的，因此，更新第i行
            for m in range(len(indexes)):
                neighbor = indexes[m]
                feature_similarities[i][neighbor] = max(0, feature_coefficients[m])
                predict_similarities[i][neighbor] = max(0, predict_coefficients[m])
    # Todo: 这些相似性是无需梯度的
    return feature_similarities.detach(), predict_similarities.detach()
