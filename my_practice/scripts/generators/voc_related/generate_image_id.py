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

# types = ["train","val"]
types = ["test"]

for type in types:
    images = dict()
    # 遍历训练集中的每张图片，创建一个字典
    main_dir = "/home/klaus125/research/dataset/VOC2007_Expanded/VOCdevkit/VOC2007/ImageSets/Main"
    type_file = os.path.join(main_dir, f"{type}.txt")
    # Todo: 先读取train/val包含的文件列表
    with open(type_file, 'r') as file:
        for line in file:
            image_id = line.strip()
            # 创建image_info
            image_info = dict()
            image_info["file_name"] = f'{image_id}.jpg'
            image_info["labels"] = []
            # 再加个名称，增强可读性
            image_info["label_names"] = []
            images[image_id] = image_info
    # 遍历每一个category的名称以及其id
    target_path = "/home/klaus125/research/fate/my_practice/dataset/voc_expanded"
    category_file = os.path.join(target_path, "category.json")
    category_path = open(category_file, 'r')
    categories = json.load(category_path)
    for category in categories:
        category_id = categories[category]
        # 读取category_train文件，读取每行数据，如果是1，则加入到对应的labels中去
        annotation_file = f'{category}_{type}.txt'
        annotation_path = os.path.join(main_dir, annotation_file)
        with open(annotation_path, 'r') as file:
            for line in file:
                image_id, val = line.split()
                # 0表示只露出了一部分，也把它加到标签集合中
                if val in {"1", "0"}:
                    images[image_id]["labels"].append(category_id)
                    images[image_id]["label_names"].append(category)
    type = "val" if type == "test" else type
    target_image_id_file = os.path.join(target_path, f"{type}_image_id.json")
    with open(target_image_id_file,'w') as json_file:
        json.dump(images, json_file)
