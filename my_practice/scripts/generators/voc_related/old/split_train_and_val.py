import json
import os
import shutil

# 划分训练集和验证集

# dataset = "voc_expanded"
dataset = "voc2012"

train_image_id_path = f"/home/klaus125/research/fate/my_practice/dataset/{dataset}/train_full_image_id.json"
val_image_id_path = f"/home/klaus125/research/fate/my_practice/dataset/{dataset}/val_full_image_id.json"

train_file = open(train_image_id_path, 'r')
train_images = json.load(train_file)

val_file = open(val_image_id_path, 'r')
val_images = json.load(val_file)

target_images_dir = '/home/klaus125/research/dataset/VOCdevkit/VOC2012'
images_dir = os.path.join(target_images_dir, "JPEGImages")
target_train_path = os.path.join(target_images_dir, "train")
if not os.path.exists(target_train_path):
    os.makedirs(target_train_path)
target_val_path = os.path.join(target_images_dir, "val")

if not os.path.exists(target_val_path):
    os.makedirs(target_val_path)
for filename in os.listdir(images_dir):
    if not filename.endswith(".jpg"):
        print("Found not jpg file.")
        continue
    image_id = filename
    source_path = os.path.join(images_dir, filename)
    target_path = None
    # 属于训练集图像
    if image_id in train_images:
        target_path = os.path.join(target_train_path, filename)
    elif image_id in val_images:
        target_path = os.path.join(target_val_path, filename)
    if target_path is not None:
        shutil.copy(source_path, target_path)


