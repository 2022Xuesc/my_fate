import os
import json

all_labels_dir = '/home/klaus125/research/dataset/NUS-WIDE/Groundtruth/AllLabels'
target_path = "/home/klaus125/research/fate/my_practice/dataset/nuswide"
category_file = os.path.join(target_path, "category.json")
category_path = open(category_file, 'r')
categories = json.load(category_path)

label_vecs = [[] for _ in range(269648)]

for category in categories:
    annotation_file = f'Labels_{category}.txt'
    annotation_path = os.path.join(all_labels_dir, annotation_file)
    with open(annotation_path, 'r') as file:
        vals = file.readlines()
        for i in range(len(vals)):
            if vals[i].strip() == "1":
                label_vecs[i].append(category)

empty_cnt = 0
for label_vec in label_vecs:
    if len(label_vec) == 0:
        empty_cnt += 1
print(f'{empty_cnt} / {len(label_vecs)}')
