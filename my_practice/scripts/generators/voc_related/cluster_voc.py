import json
import os

import shutil
from kmodes import kmodes
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def get_label_vecs(json_path, num_labels=20):
    vec2names = dict()
    # 训练数据集的信息，从train_anno.json中读取
    image_id2labels = json.load(open(json_path, 'r'))
    for image_id in image_id2labels:
        image_info = image_id2labels[image_id]
        filename = image_info['file_name']
        label = image_info['labels']
        # 将label转换成80维度的向量
        label_vec = [0] * num_labels
        for label_index in label:
            label_vec[label_index] = 1
        label_key = tuple(label_vec)
        if label_key not in vec2names.keys():
            vec2names[label_key] = list()
        vec2names[label_key].append(filename)
    vecs = []
    for key in vec2names.keys():
        vecs.append(list(key))
    return vec2names, vecs


def copy_file_to_cluster(src_dir, clustered_dir, clusters, data, phase, vec2names, vecs):
    for i in range(data.shape[0]):
        target_dir = os.path.join(clustered_dir, f'client{clusters[i] + 1}/{phase}')
        os.makedirs(target_dir, exist_ok=True)
        for filename in vec2names[tuple(vecs[i])]:
            # 将filename对应的文件拷贝到target_path中
            shutil.copy(os.path.join(src_dir, filename), target_dir)


# dataset = "voc"
dataset = "voc_expanded"

train_image_id_path = f"/home/klaus125/research/fate/my_practice/dataset/{dataset}/train_image_id.json"
val_image_id_path = f"/home/klaus125/research/fate/my_practice/dataset/{dataset}/val_image_id.json"

# 仍然选择划分10个客户端
num_clients = 10
# train_dir需要修改
train_dir = "/home/klaus125/research/dataset/VOC2007/JPEGImages/origin"
val_dir = "/home/klaus125/research/dataset/VOC2007_Expanded/VOCdevkit/VOC2007/JPEGImages"
clustered_dir = "/home/klaus125/research/dataset/VOC2007_Expanded/clustered_voc"

# 使用kmodes聚类方法
km = kmodes.KModes(n_clusters=num_clients)
# 根据train_image_id获取标签向量
# coco数据集的标签数量是20
train_vec2names, train_vecs = get_label_vecs(train_image_id_path, num_labels=20)
train_data = np.array(train_vecs)
print(f'训练数据的维度为：{train_data.shape}')
print('开始训练聚类模型并预测')
train_clusters = km.fit_predict(train_data)
print('训练完成')
copy_file_to_cluster(train_dir, clustered_dir, train_clusters, train_data,
                     'train', train_vec2names, train_vecs)

val_vec2names, val_vecs = get_label_vecs(val_image_id_path)
val_data = np.array(val_vecs)
val_clusters = km.predict(val_data)
copy_file_to_cluster(val_dir, clustered_dir, val_clusters, val_data, 'val', val_vec2names, val_vecs)
