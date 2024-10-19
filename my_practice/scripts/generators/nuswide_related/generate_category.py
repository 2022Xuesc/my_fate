# 创建字典
import json
import os

categories = dict()
# 标签的编号从0开始
index = 0
concept_filename = "/home/klaus125/research/dataset/NUS-WIDE/Concepts81.txt"
concept_file = open(concept_filename,'r')
# 遍历dir_name下的每个文件
for category in concept_file.readlines():
    category = category.strip()
    # 如果category不在字典中，则将其加入
    if category not in categories:
        categories[category] = index
        index += 1
save_dir = "/home/klaus125/research/fate/my_practice/dataset/nuswide"
file_path = os.path.join(save_dir, "category.json")
with open(file_path, 'w') as json_file:
    json.dump(categories, json_file, indent=2)
