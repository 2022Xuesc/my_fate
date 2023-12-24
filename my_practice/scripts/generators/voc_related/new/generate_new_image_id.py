# 遍历trainval和test，生成一个file_name到20维的labels的map，标签中包含0、-1和1，然后手动处理
import os.path
import csv
import numpy
import json


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


csv_dir = '/home/klaus125/research/dataset/voc_standalone/csvs'
types = ['trainval', 'test']
for type in types:
    file_csv = os.path.join(csv_dir, 'classification_' + type + '.csv')
    image2infos = read_object_labels_csv(file_csv)
    # Todo: 对image2infos进行持久化
    type = "val" if type == "test" else 'train'
    target_image_id_file = os.path.join('.', f"{type}_full_image_id.json")
    with open(target_image_id_file, 'w') as json_file:
        json.dump(image2infos, json_file)
