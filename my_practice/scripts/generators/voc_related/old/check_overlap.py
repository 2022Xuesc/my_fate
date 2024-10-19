import os

import json

# Todo: 检查train/val_image_id.json生成的正确性
train_image_id_path = "/home/klaus125/research/fate/my_practice/dataset/voc2012/train_full_image_id.json"
val_image_id_path = "/home/klaus125/research/fate/my_practice/dataset/voc2012/val_full_image_id.json"

train_images = '/home/klaus125/research/dataset/VOCdevkit/VOC2012/train'
val_images = '/home/klaus125/research/dataset/VOCdevkit/VOC2012/val'

print('训练数据集大小', len(os.listdir(train_images)))
print('验证数据集大小', len(os.listdir(val_images)))


train_file = open(train_image_id_path, 'r')
train_images = json.load(train_file)
print("训练集大小： ", len(train_images))
val_file = open(val_image_id_path, 'r')
val_images = json.load(val_file)
print("验证集大小： ", len(val_images))

train_list_file = '/home/klaus125/research/dataset/VOCdevkit/VOC2012/ImageSets/Main/train.txt'
val_list_file = '/home/klaus125/research/dataset/VOCdevkit/VOC2012/ImageSets/Main/val.txt'

train_list = open(train_list_file)
cnt = 0
for train_image in train_list:
    cnt += 1
print(cnt)

# 将两个字典合并输出为train_image_id_path
# print(len(train_images))
# train_images.update(val_images)
# target_path = "/home/klaus125/research/fate/my_practice/dataset/voc_expanded/train_image_id.json"
# with open(target_path,'w') as json_file:
#     json.dump(train_images,json_file)


# 判断键集合是否有重叠
if set(train_images.keys()) & set(val_images.keys()):  # 使用 & 运算符来查看两个集合的交集
    print("字典的键集合有重叠")
else:
    print("字典的键集合没有重叠")

# Todo: 检查训练集和测试集划分的正确性
# train_path = "/home/klaus125/research/dataset/VOC2007/JPEGImages/train"
# val_path = "/home/klaus125/research/dataset/VOC2007/JPEGImages/val"
# print("训练集大小", len(os.listdir(train_path)))
# print("验证集大小", len(os.listdir(val_path)))


# Todo: 输出一下聚类后每个客户端的数据集大小
num_clients = 10
clustered_dir = "/home/klaus125/research/dataset/VOCdevkit/VOC2012/clustered_voc"
for i in range(1,num_clients + 1):
    client_path = os.path.join(clustered_dir,f'client{i}')
    client_train_path = os.path.join(client_path,'train')
    client_val_path = os.path.join(client_path,'val')
    print(f"===============客户端{i}=================")
    print("训练集大小", len(os.listdir(client_train_path)))
    print("验证集大小", len(os.listdir(client_val_path)))
