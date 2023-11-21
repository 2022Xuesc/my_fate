import os

import json
import shutil

# 划分训练集和验证集

# dataset = "voc_expanded"
dataset = "voc"

train_image_id_path = f"/home/klaus125/research/fate/my_practice/dataset/{dataset}/train_image_id.json"
val_image_id_path = f"/home/klaus125/research/fate/my_practice/dataset/{dataset}/val_image_id.json"

train_file = open(train_image_id_path, 'r')
train_images = json.load(train_file)

val_file = open(val_image_id_path, 'r')
val_images = json.load(val_file)

images_dir = "/home/klaus125/research/dataset/VOC2007/JPEGImages"
target_train_path = os.path.join(images_dir, "train")
if not os.path.exists(target_train_path):
    os.makedirs(target_train_path)
target_val_path = os.path.join(images_dir, "val")

if not os.path.exists(target_val_path):
    os.makedirs(target_val_path)
for filename in os.listdir(images_dir):
    if not filename.endswith(".jpg"):
        continue
    image_id = filename.split(".")[0]
    source_path = os.path.join(images_dir, filename)
    target_path = None
    # 属于训练集图像
    if image_id in train_images:
        target_path = os.path.join(target_train_path, filename)
    else:
        target_path = os.path.join(target_val_path, filename)
    shutil.move(source_path, target_path)
