import json
import os
import shutil

dir_name = "/home/klaus125/research/dataset/val2014"
target_dir = "/home/klaus125/research/fate/my_practice/knowledge/cam/my_imgs"
json_path = "val_image_id.json"
val_json = json.load(open(json_path, 'r'))
person_images = set()
for image_id in val_json:
    item = val_json[image_id]
    label = item['labels']
    if 49 in label:
        person_images.add(item['file_name'])
# cnt = 10
# 遍历dir_name下每张图片
files = os.listdir(dir_name)
for file_name in files:
    if file_name in person_images:
        shutil.copy(os.path.join(dir_name, file_name), target_dir)
        # cnt -= 1
        # if cnt == 0:
        #     break
