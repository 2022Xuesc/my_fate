import json
import os
import shutil

remove_labels = [5, 13, 14, 18, 26, 27, 39, 41, 42, 47, 49, 57, 62, 75]

dir_name = "/home/klaus125/research/dataset/val2014"
target_dir = "/home/klaus125/research/fate/my_practice/knowledge/cam/my_imgs"
json_path = "val_image_id.json"
val_json = json.load(open(json_path, 'r'))

# 遍历每一个标签
for remove_label in remove_labels:
    label_images = set()
    for image_id in val_json:
        item = val_json[image_id]
        label = item['labels']
        if remove_label in label:
            label_images.add(item['file_name'])
    target_path = os.path.join(target_dir, str(remove_label))
    # 如果target_path指向目录不存在，则创建目录
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    # Todo: 移动图片的数目
    cnt = 10
    # 遍历dir_name下每张图片
    files = os.listdir(dir_name)
    for file_name in files:
        if file_name in label_images:
            shutil.copy(os.path.join(dir_name, file_name), target_path)
            cnt -= 1
            if cnt == 0:
                break

