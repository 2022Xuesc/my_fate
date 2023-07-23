import json
import numpy as np
import pickle

# 分析训练集中的数据
image_id2labels = json.load(open('train_image_id.json', 'r'))
adjList = np.zeros((80, 80))
for image_id in image_id2labels:
    image_info = image_id2labels[image_id]
    label = image_info['labels']
    n = len(label)
    for i in range(n):
        for j in range(i + 1, n):
            x = label[i]
            y = label[j]
            adjList[x][y] += 1
            adjList[y][x] += 1
# # 和pkl进行校验
# print('Done')

adj_file = '/home/klaus125/research/fate/my_practice/dataset/coco/coco_adj.pkl'
result = pickle.load(open(adj_file, 'rb'))
_adj = result['adj']
_nums = result['nums']
_nums = _nums[:, np.newaxis]
# 转换成条件概率的形式
_adj = _adj / _nums
# Todo: 根据邻接矩阵给出类的从属关系
print("Todo")
