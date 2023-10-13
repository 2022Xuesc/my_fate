import json
import numpy as np

num_labels = 20
json_file = '/home/klaus125/research/dataset/VOC2007_Expanded/clustered_voc/client3/train/anno.json'
image_id2labels = json.load(open(json_file, 'r'))
adjList = np.zeros((num_labels, num_labels))
nums = np.zeros(num_labels)
for image_info in image_id2labels:
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
# 遍历每一行
for i in range(num_labels):
    if nums[i] != 0:
        adjList[i] = adjList[i] / nums[i]
# 遍历A，将主对角线元素设置为1
for i in range(num_labels):
    adjList[i][i] = 1
print("Hello")
