import os

import json
import shutil

# 划分训练集和验证集

dataset = "nuswide"

images_dir = "/home/klaus125/research/dataset/NUS-WIDE/images"
train_txt = '/home/klaus125/research/dataset/NUS-WIDE/ImageList/TrainImagelist.txt'
valid_txt = '/home/klaus125/research/dataset/NUS-WIDE/ImageList/TestImagelist.txt'

train_images = open(train_txt, 'r').readlines()
train_set = set([train_image.split('\\')[1].strip() for train_image in train_images])
valid_images = open(valid_txt, 'r').readlines()
valid_set = set([valid_image.split('\\')[1].strip() for valid_image in valid_images])

print('训练集数据大小', len(train_set))
print('验证集数据大小', len(valid_set))

target_train_path = os.path.join(images_dir, "train")
if not os.path.exists(target_train_path):
    os.makedirs(target_train_path)
    
target_val_path = os.path.join(images_dir, "val")
if not os.path.exists(target_val_path):
    os.makedirs(target_val_path)

for filename in os.listdir(images_dir):
    if not filename.endswith(".jpg"):
        print(filename)
        continue
    source_path = os.path.join(images_dir, filename)
    # 判断属于训练集还是验证集
    if filename in train_set:
        target_path = os.path.join(target_train_path, filename)
    elif filename in valid_set:
        target_path = os.path.join(target_val_path, filename)
    else:
        print("ERROR")
        break
    shutil.move(source_path, target_path)
