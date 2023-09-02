import pickle
import numpy as np

# file_name = '../data/statistics.pkl'
file_name = 'server_statistics.pkl'
file = open(file_name, 'rb')
data = pickle.load(file)

# complement_labels = [35,69]
# Todo: 有两个不充足标签(分别加到索引35和索引69的位置上)
# complement_data = pickle.load(open('server_statistics.pkl', 'rb'))[:2]
# data.insert(complement_labels[0], complement_data[0])
# data.insert(complement_labels[1], complement_data[1])


arr = np.transpose(np.array(data))
remove_labels = {5, 13, 14, 18, 26, 27, 39, 41, 42, 47, 49, 57, 62, 75}

num_labels = 80
num_layers = len(arr)
# 遍历每一层
for i in range(num_layers):
    # 内部是num_labels * num_channels
    num_channels = len(arr[i][0])
    save_cam = np.zeros(num_channels)
    drop_cam = np.zeros(num_channels)
    # 遍历每个标签，如果是保留标签，则加到save_cam上，否则加到drop_cam上
    for label in range(num_labels):
        if label in remove_labels:
            drop_cam += arr[i][label]
        else:
            save_cam += arr[i][label]
    # 统计完毕后，从drop_cam和save_cam选出ratio比例的通道，计算其IOU值
    ratio = 0.1
    k = int(num_channels * ratio)
    save_channels = np.argsort(save_cam)[-k:]
    drop_channels = np.argsort(drop_cam)[-k:]
    # 计算交集和并集
    intersection = np.intersect1d(save_channels, drop_channels)
    union = np.union1d(save_channels, drop_channels)

    # 计算交并比（Jaccard Index）
    jaccard_index = len(intersection) / len(union)

    # print(f"Intersection: {intersection}")
    # print(f"Union: {union}")
    print(f"layer:{i},  num_channels: {num_channels},  Jaccard Index: {jaccard_index:.4f}")
    # print(--)
