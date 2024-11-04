import os

from dataset_generator import *

train_path = "/data/projects/dataset/train2017"
val_path = "/data/projects/dataset/val2017"
test_path = "/data/projects/dataset/test2017"


# train_path = "../../my_data/imagenet/guest/train/images"
# val_path = "../../my_data/imagenet/guest/train/images"
# test_path = "../../my_data/imagenet/guest/train/images"

coco_dir =  '../../../dataset/coco'
def generate_dataset(n):
    my_data_path = "/data/projects/iid_dataset"
    for i in range(1, n + 1):
        client_path = os.path.join(my_data_path, f'client{i}')
        client_train_path = os.path.join(client_path, 'train')
        client_val_path = os.path.join(client_path, 'val')
        # client_test_path = os.path.join(client_path, 'test')
        generate_2014(train_path, (i - 1) / n, i / n, client_train_path)
        generate_2014(val_path, (i - 1) / n, i / n, client_val_path)
        # generate_2017(test_path, (i - 1) / n, i / n, client_test_path)
        # 生成标签
        generate_anno(coco_dir,client_train_path,'train')
        generate_anno(coco_dir,client_val_path,'val')
        generate_configs([client_train_path, client_val_path])


# 先跳过测试数据集的划分
def generate_labels_for_clusters(dir_path, client_num):
    for i in range(1, client_num + 1):
        client_path = os.path.join(dir_path, f'client{i}')
        client_train_path = os.path.join(client_path, 'train')
        client_val_path = os.path.join(client_path, 'val')
        path_list = [client_train_path, client_val_path]
        generate_labels(path_list)
        generate_configs(path_list)


def generate_embedding_labels_for_clusters(dir_path, client_num):
    for i in range(1, client_num + 1):
        client_path = os.path.join(dir_path, f'client{i}')
        client_train_path = os.path.join(client_path, 'train')
        client_val_path = os.path.join(client_path, 'val')
        path_list = [client_val_path]
        generate_embedding_labels(path_list)
        generate_configs(path_list)


# 生成全局的标注数据
annotations_dir = "/data/projects/dataset/coco2017/annotations"
train_annotation_file = os.path.join(annotations_dir, 'instances_train2017.json')
val_annotation_file = os.path.join(annotations_dir, 'instances_val2017.json')
# test_annotation_file = os.path.join(annotations_dir, 'image_info_test2017.json')

save_image2labels(train_annotation_file, 'train')
save_image2labels(val_annotation_file, 'val')
# Todo: 先跳过对测试集的处理
# save_image2labels(test_annotation_file, 'test')

# 为每个文件夹生成标注数据以及config.yaml文件
# generate_dataset(8)

# clustered_dir = '/data/projects/clustered_dataset'
# generate_embedding_labels_for_clusters(clustered_dir, client_num=8)

client_nums = 10
generate_dataset(client_nums)
