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
prob = nums / len(image_id2labels)
nums = nums[:, np.newaxis]
# 遍历每一行
for i in range(num_labels):
    if nums[i] != 0:
        adjList[i] = adjList[i] / nums[i]
# 遍历A，将主对角线元素设置为1
for i in range(num_labels):
    adjList[i][i] = 1

adjList2 = np.zeros((num_labels, num_labels))
# 记录i未出现的次数
not_occurred_nums = np.zeros(num_labels)
for image_info in image_id2labels:
    labels = image_info['labels']
    for i in range(num_labels):
        # 如果该图像中不包含标签label
        if i not in labels:
            not_occurred_nums[i] += 1
            for j in labels:
                adjList2[i][j] += 1
not_occurred_nums = not_occurred_nums[:, np.newaxis]
# 验证adjList2的每一行是否和为1
# 注意，adjList2本来就不为1
for i in range(num_labels):
    if not_occurred_nums[i] != 0:
        adjList2[i] = adjList2[i] / not_occurred_nums[i]
print("Hello")

# Todo: 同时，导出标签j出现的概率，用于判断是促进作用还是抑制作用
np.save('label_occur_prob_vec',prob)
arr = np.load('label_occur_prob_vec.npy')