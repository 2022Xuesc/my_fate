import json
import numpy as np

import os.path
import shutil

# images_path = 'coco/val_image_id.json'
# images = json.load(open(images_path, 'r'))
# # print(images)
# 
# cnt1 = 0
# cnt2 = 0
# # 找出同时具有32和27的图像名称，找出只具有32的图像名称
# for image_id in images:
#     image_info = images[image_id]
#     labels = image_info['labels']
#     label_set = set(labels)
#     if 32 in label_set and 27 in label_set:
#         if cnt1 != 20:
#            print(image_info['file_name'])
#            cnt1 += 1
#     if 32 in label_set and 27 not in label_set:
#         if cnt2 == 0:
#             print(image_info['file_name'])
#             cnt2 += 1
#             
anno_dir = '/home/klaus125/research/fate/my_practice/dataset/voc_expanded'
image_id2labels = json.load(open(os.path.join(anno_dir, 'train_full_image_id.json'), 'r'))
num_labels = 20


adjList = np.zeros((num_labels, num_labels))
# 还要维护单标签出现的次数
nums = np.zeros(num_labels)
for image_id in image_id2labels:
    labels = image_id2labels[image_id]['labels']
    true_labels = []
    # 1. 找出0和1的标签id，然后使用相同的api进行计算
    for label_id in range(num_labels):
        if labels[label_id] == 0 or labels[label_id] == 1:
            true_labels.append(label_id)
            nums[label_id] += 1
    n = len(true_labels)
    for i in range(n):
        for j in range(i + 1, n):
            x = true_labels[i]
            y = true_labels[j]
            adjList[x][y] += 1
            adjList[y][x] += 1
nums = nums[:, np.newaxis]
# 遍历每一行
for i in range(num_labels):
    if nums[i] != 0:
        adjList[i] = adjList[i] / nums[i]

print("hello world")


# 19_17
cnt1 = 0
cnt2 = 0
# # 找出同时具有19和17的图像名称，找出只具有19的图像名称
image_dir = '/home/klaus125/research/dataset/voc_standalone/trainval'
for image_id in image_id2labels:
    labels = image_id2labels[image_id]['labels']
    label_set = set()
    for i in range(num_labels):
        if labels[i] == 1:
            label_set.add(i)
    print(label_set)
    if 19 in label_set and 17 in label_set:
        if cnt1 != 1:
            # print(os.path.join(anno_dir, file_name))
            shutil.copy(os.path.join(image_dir, image_id), f'typical_images/voc/both_{cnt1}.jpg')
            cnt1 += 1
    if 19 in label_set and 17 not in label_set:
        if cnt2 != 3:
            shutil.copy(os.path.join(image_dir, image_id), f'typical_images/voc/only_tv_{cnt2}.jpg')
            cnt2 += 1
