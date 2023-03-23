import os

import torch
import shutil
from kmodes import kmodes
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def train_cluster(data, start_k=2, end_k=20):
    # 记录训练过程中sse关于k值的变化情况
    K = []
    SSE = []
    silhouette_all = []
    # 保存模型
    models = []
    for i in range(start_k, end_k):
        # 获取聚类数为i时的聚类模型
        kmodes_model = kmodes.KModes(n_clusters=i, n_jobs=multiprocessing.cpu_count())
        # 与给定数据进行拟合
        kmodes_model.fit(data)
        # 计算拟合数据与模型输出的轮廓分数
        a = silhouette_score(data, kmodes_model.labels_, metric='hamming')
        # 保存每个k值对应的SSE值
        SSE.append(kmodes_model.cost_)
        K.append(i)
        print('{} Means SSE loss = {}'.format(i, kmodes_model.cost_))
        silhouette_all.append(a)
        print('这个是k={}次时的轮廓系数{}：'.format(i, a))
        # 保存每个k值对应的模型
        models.append(kmodes_model)
    return K, SSE, silhouette_all, models


def learn_kmodes():
    # 从[1,100)中取出维度为(100,10)的随机数
    data = np.random.choice(100, (100, 10))
    # 对于不同的k，训练模型，输出cost、sse以及训练好的模型
    res = train_cluster(data)
    K = res[0]
    # SSE = res[1]
    # plt.plot(K, SSE, 'bx-')
    # plt.xlabel('聚类类别数k')
    # plt.ylabel('SSE')
    # plt.xticks(K)
    # plt.title('用肘部法则来确定最佳的k值')
    # plt.show()

    # Todo: 训练好模型后，取出模型进行预测
    k = 5
    best_model = res[3][K.index(k)]
    # 预测模型
    clusters = best_model.predict(data)
    centroids = best_model.cluster_centroids_


def draw(metric_type, K, metrics):
    plt.plot(K, metrics, 'bx-')
    plt.xlabel('聚类类别数k')
    plt.ylabel(metric_type)
    plt.xticks(K)
    plt.title(f'the {metric_type} curve on k')
    plt.show()


# learn_kmodes()


def get_label_vecs(data_dir, num_classes=90):
    vec2names = dict()
    # 为train.txt也生成labels.txt
    labels_path = os.path.join(data_dir, 'labels.txt')
    fp = open(labels_path, 'r')
    for line in fp:
        label_vec = [0] * num_classes
        line.strip('\n')
        info = line.split(',')
        for index in range(1, len(info)):
            if info[index] == '1':
                label_vec[index - 1] = 1
        label_key = tuple(label_vec)
        if label_key not in vec2names.keys():
            vec2names[label_key] = list()
        # info[0]是文件名
        vec2names[label_key].append(info[0])

    vecs = []
    for key in vec2names.keys():
        vecs.append(list(key))
    return vec2names, vecs


def copy_file_to_cluster(src_dir, clustered_dir, clusters, data, phase, vec2names, vecs):
    for i in range(data.shape[0]):
        target_dir = os.path.join(clustered_dir, f'client{clusters[i]}/{phase}')
        os.makedirs(target_dir, exist_ok=True)
        for filename in vec2names[tuple(vecs[i])]:
            # 将filename对应的文件拷贝到target_path中
            shutil.copy(os.path.join(src_dir, filename), target_dir)


# 训练得到损失曲线，从曲线中选择最好的k
# res = train_cluster(data)
# K = res[0]
# draw('dist',K,res[1])
# draw('sse', K, res[2])

num_clients = 2
# 1. 读取训练数据集，根据标签之间的距离对数据集进行聚类
train_dir = '/data/projects/dataset/train2014'
val_dir = '/data/projects/dataset/val2014'
clustered_dir = '/data/projects/clustered_dataset'

km = kmodes.KModes(n_clusters=num_clients)
train_vec2names, train_vecs = get_label_vecs(train_dir)
# 转为array数组
train_data = np.array(train_vecs)
print(f'训练数据的维度为：{train_data.shape}')
print('开始训练聚类模型并预测')
train_clusters = km.fit_predict(train_data)
print('训练完成')
copy_file_to_cluster(train_dir, clustered_dir, train_clusters, train_data,
                     'train', train_vec2names, train_vecs)

# 2. 根据聚类结果，划分训练数据集，并且对验证数据集进行预测

val_vec2names, val_vecs = get_label_vecs(val_dir)
val_data = np.array(val_vecs)
val_clusters = km.predict(val_data)
copy_file_to_cluster(val_dir, clustered_dir, val_clusters, val_data, 'val', val_vec2names, val_vecs)
