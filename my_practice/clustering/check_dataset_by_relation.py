import json
import numpy as np


def get_entropy(p):
    entropy = 0
    n = len(p)
    for i in range(n):
        # 如果概率值为0，则不进行计算
        if p[i] == 0:
            continue
        entropy += -p[i] * np.log2(p[i])
    return entropy


adjLists = []
# Todo: 初始化json_file
json_files = []
num_labels = 80
num_clients = 10
relation_th = 0.5
entropy_th = 1

for i in range(num_clients):
    json_files.append(f'/data/projects/clustered_dataset/client{i + 1}/train/anno.json')

for json_file in json_files:
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
    adjLists.append(adjList)

# 遍历每个标签对
entropies = []
pList = []
for i in range(num_labels):
    for j in range(i + 1, num_labels):
        p = np.zeros(4)
        # 遍历每个客户端
        for k in range(num_clients):
            i2j = adjLists[k][i][j] > relation_th
            j2i = adjLists[k][j][i] > relation_th
            # 根据i2j和j2i确定客户端所属的类别
            p[i2j * 2 + j2i] += 1
        p /= np.sum(p)
        pList.append(p)
        # 把概率值也存下来
        h = get_entropy(p)
        entropies.append(h)
        print(f'the entropy of ({i},{j}) is {h}')
# Todo: 这里观察每个标签对的熵
# 把概率向量存储下来
file_name = 'p_list.npy'
np.save(file_name, pList)
# 把熵存储下来
