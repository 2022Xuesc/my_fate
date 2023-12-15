import json
import numpy as np
import pickle

# 分析训练集中的数据
image_id2labels = json.load(open('val_image_id.json', 'r'))

# ls = []
# for image_id in image_id2labels:
#     image_info = image_id2labels[image_id]
#     ls.append(image_info)
# with open('anno2.json','w') as json_file:
#     json.dump(ls,json_file)

adjList = np.zeros((80, 80))
# 还要维护单标签出现的次数
nums = np.zeros(80)
for image_id in image_id2labels:
    image_info = image_id2labels[image_id]
    labels = image_info['labels']
    for label in labels:
        nums[label] += 1
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            x = labels[i]
            y = labels[j]
            adjList[x][y] += 1
            adjList[y][x] += 1
nums = nums[:, np.newaxis]
for i in range(len(adjList)):
    if nums[i] != 0:
        adjList[i] = adjList[i] / nums[i]

print("Hello World")
#
# adj_file = '/home/klaus125/research/fate/my_practice/dataset/coco/coco_adj.pkl'
# result = pickle.load(open(adj_file, 'rb'))
# _adj = result['adj']
# _nums = result['nums']
# _nums = _nums[:, np.newaxis]
# # 转换成条件概率的形式
# _adj = _adj / _nums
# # # Todo: 根据邻接矩阵给出类的从属关系
# print("Todo")
