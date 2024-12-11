# 遍历trainval和test，生成一个file_name到20维的labels的map，标签中包含0、-1和1，然后手动处理
import csv
import json
import os.path


def read_object_labels_csv(file, header=True):
    image2infos = dict()
    num_categories = 0
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0] + ".jpg"
                labels = [int(label) for label in row[1:num_categories + 1]]
                image_info = {'file_name': name, 'labels': labels}
                image2infos[name] = image_info
            rownum += 1
    return image2infos


category_dir = '/home/klaus125/research/fate/my_practice/dataset/voc2012/category.json'
dataset_dir = '/home/klaus125/research/dataset/VOCdevkit/VOC2012/ImageSets/Main'
types = ['train', 'val']
categories = None
with open(category_dir, 'r', encoding='utf-8') as file:
    categories = json.load(file)    

for type in types:
    # 读取训练集/验证集中的每个图像，存储到map中，对应一个长度为20的标签向量
    type_total_file = os.path.join(dataset_dir, f'{type}.txt')
    f = open(type_total_file)
    image_ids = dict()
    for line in f:
        file_name = line.strip() + '.jpg'
        image_ids[file_name] = {
            'file_name': file_name,
            'labels': [0 for _ in range(20)]
        }
    for category_name in categories:
        category_id = categories[category_name]
        label_file = os.path.join(dataset_dir, f'{category_name}_{type}.txt')
        label_f = open(label_file)
        for line in label_f:
            words = line.strip().split(' ')
            file_name = words[0] + '.jpg'
            label_tag = int(words[-1])
            image_ids[file_name]['labels'][category_id] = label_tag
            if label_tag == 0:
                print("Found you!")
    # Todo: 将该类型的image_ids进行持久化
    target_image_id_file = os.path.join('.', f"{type}_full_image_id.json")
    # with open(target_image_id_file, 'w') as json_file:
    #     json.dump(image_ids, json_file)
