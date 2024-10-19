# 生成train_image_id.json和val_image_id.json
'''
基本格式: dict()字典
"57870": {
    "file_name": "COCO_train2014_000000057870.jpg",
    "labels": [12, 77, 51, 22, 27]
}
'''
import json
import os
import shutil

types = ["train", "test"]

for type in types:
    images = dict()
    # 遍历训练集中的每张图片，创建一个字典
    main_dir = "/home/klaus125/research/dataset/NUS-WIDE/ImageList"
    label_info_dir = '/home/klaus125/research/dataset/NUS-WIDE/Groundtruth/TrainTestLabels'
    Type = type.capitalize()
    type_file = os.path.join(main_dir, f"{Type}Imagelist.txt")
    image_ids = []
    with open(type_file, 'r') as file:
        for line in file:
            image_id = line.split('\\')[1].strip().split('.')[0]
            image_ids.append(image_id)
            # 创建image_info
            image_info = dict()
            image_info["file_name"] = f'{image_id}.jpg'
            image_info["labels"] = []
            # 再加个名称，增强可读性
            image_info["label_names"] = []
            images[image_id] = image_info
    # 遍历每一个category的名称以及其id
    target_path = "/home/klaus125/research/fate/my_practice/dataset/nuswide"
    category_file = os.path.join(target_path, "category.json")
    category_path = open(category_file, 'r')
    categories = json.load(category_path)
    for category in categories:
        category_id = categories[category]
        # 读取category_train文件，读取每行数据，如果是1，则加入到对应的labels中去
        annotation_file = f'Labels_{category}_{Type}.txt'
        annotation_path = os.path.join(label_info_dir, annotation_file)
        with open(annotation_path, 'r') as file:
            vals = file.readlines()
            for i in range(len(vals)):
                image_id = image_ids[i]
                val = vals[i].strip()
                if val == '1':
                    images[image_id]["labels"].append(category_id)
                    images[image_id]["label_names"].append(category)
    type = "val" if type == "test" else type
    # Todo: 为什么有这么多标记为空的样本
    # 遍历每张图片，如果没有标注的标签，则移动到unlabeled/{type}文件夹下
    image_dir = '/home/klaus125/research/dataset/NUS-WIDE/images'
    removed_keys = []
    for image_id in images:
        image_info = images[image_id]
        file_name = image_info['file_name']
        if len(image_info['labels']) == 0:
            removed_keys.append(image_id)
            # Todo: 将空标签图片移到其他文件夹下
            # source_path = os.path.join(os.path.join(image_dir, type), file_name)
            # unlabeled_path = os.path.join(f'{image_dir}/unlabeled/{type}', file_name)
            # shutil.move(source_path, unlabeled_path)
    [images.pop(removed_key) for removed_key in removed_keys]
    print(f'type = {type}, size = {len(images)}')
    target_image_id_file = os.path.join(target_path, f"{type}_image_id.json")
    with open(target_image_id_file, 'w') as json_file:
        json.dump(images, json_file)
