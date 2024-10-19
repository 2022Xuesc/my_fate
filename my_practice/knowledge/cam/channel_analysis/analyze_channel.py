import pickle
import numpy as np
import matplotlib.pyplot as plt

arr = np.array([[1, 2], [3, 4]])
print(arr)

# file_name = '../data/statistics.pkl'
file_name = 'server_statistics.pkl'
file = open(file_name, 'rb')
data = pickle.load(file)

complement_labels = [35, 69]
label_img_cnts = {35: 70, 69: 74}
# Todo: 有两个不充足标签(分别加到索引35和索引69的位置上)
complement_data = pickle.load(open('complement_statistics.pkl', 'rb'))[:2]
data.insert(complement_labels[0], complement_data[0])
data.insert(complement_labels[1], complement_data[1])

arr = np.transpose(np.array(data))
remove_labels = {5, 13, 14, 18, 26, 27, 39, 41, 42, 47, 49, 57, 62, 75}

num_labels = 80
num_layers = len(arr)
# 遍历每一层

ratio = 0.9
import_ratios = []
for i in range(num_layers):
    # 内部是num_labels * num_channels
    num_channels = len(arr[i][0])
    save_cam = np.zeros(num_channels)
    drop_cam = np.zeros(num_channels)
    # 遍历每个标签，如果是保留标签，则加到save_cam上，否则加到drop_cam上
    # Todo: 有的标签不够80张图像，因此，需要使用图像张数作为权值
    img_cnts = 0
    if i in label_img_cnts:
        img_cnts = label_img_cnts[i]
    else:
        img_cnts = 100
    for label in range(num_labels):
        incr = arr[i][label]
        if label in remove_labels:
            drop_cam += arr[i][label] / img_cnts
        else:
            save_cam += arr[i][label] / img_cnts
    # 统计完毕后，从drop_cam和save_cam选出ratio比例的通道，计算其IOU值
    k = int(num_channels * ratio)
    save_channels = np.argsort(save_cam)[-k:]
    drop_channels = np.argsort(drop_cam)[-k:]
    # 计算交集和并集
    intersection = np.intersect1d(save_channels, drop_channels)
    union = np.union1d(save_channels, drop_channels)

    # 计算交并比（Jaccard Index）
    jaccard_index = len(intersection) / len(union)

    # 计算通道交集占选出通道数的比例
    import_index = (k - len(intersection)) / k
    import_ratios.append(import_index)
    # print(f"Intersection: {intersection}")
    # print(f"Union: {union}")
    # Todo: 每一层打印总通道数目、选出的通道数目以及通道交集占选出通道数的比例
    print(f"layer:{i},  num_channels: {num_channels}, selected_channels: {k}, ratio: {import_index:.4f}")
    # print(--)

save_dir = '.'
plt.figure(figsize=(12, 12))
plt.bar(range(100), import_ratios)
plt.ylim(0, 1)
plt.title('The distribution of insignificant ratios')
plt.xlabel('layer')
plt.ylabel('Insignificant ratio')
plt.savefig(f'{save_dir}/Insignificant_{ratio}.svg', dpi=600, format='svg')

