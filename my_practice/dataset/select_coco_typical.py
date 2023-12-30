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
anno_dir = '/home/klaus125/research/fate/my_practice/dataset/coco'
image_id2labels = json.load(open(os.path.join(anno_dir, 'train_image_id.json'), 'r'))

num_labels = 80

adjList = np.zeros((num_labels, num_labels))
# 还要维护单标签出现的次数
nums = np.zeros(num_labels)
for image_id in image_id2labels:
    labels = image_id2labels[image_id]['labels']
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
