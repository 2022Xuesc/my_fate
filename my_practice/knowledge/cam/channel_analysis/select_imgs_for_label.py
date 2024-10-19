import json
import os
import shutil

num_labels = 80

# Todo: 需要移除的标签
# remove_labels = [5, 13, 14, 18, 26, 27, 39, 41, 42, 47, 49, 57, 62, 75]

all_labels = list(range(num_labels))

#dir_name = "/home/klaus125/research/dataset/val2014"
dir_name = "/data/projects/dataset/val2014"
#target_dir = "/home/klaus125/research/dataset/label_imgs"
target_dir = "/data/projects/dataset/label_imgs"
json_path = "../val_image_id.json"
val_json = json.load(open(json_path, 'r'))

# Todo: 遍历每一个val_json中的每一项，建立从file_name到labels的映射
image_labels = dict()

for image_id in val_json:
    item = val_json[image_id]
    # 文件名以及对应的标签向量
    file_name = item['file_name']
    labels = item['labels']
    image_labels[file_name] = labels

# 创建文件夹
for label in range(num_labels):
    target_path = os.path.join(target_dir, str(label))
    if not os.path.exists(target_path):
        os.makedirs(target_path)

label_cnts = [0] * num_labels
max_cnt = 100
files = os.listdir(dir_name)
for file_name in files:
    # Todo: 为什么没有呢？
    if file_name not in image_labels:
        continue
    # 遍历包含的每一个标签
    for label in image_labels[file_name]:
        # 如果移动照片数量==maxCnt，则跳过
        if label_cnts[label] == max_cnt:
            continue
        shutil.copy(os.path.join(dir_name,file_name),os.path.join(target_dir, str(label)))
        label_cnts[label] += 1


