from dataset_generator import *

train_path = "/data/projects/dataset/train2014"
val_path = "/data/projects/dataset/val2014"
test_path = "/data/projects/dataset/test2014"


def generate_dataset(n):
    my_data_path = "/data/projects/my_dataset"
    for i in range(n + 1):
        client_path = os.path.join(my_data_path, f'client{i}')
        client_train_path = os.path.join(client_path, 'train')
        client_val_path = os.path.join(client_path, 'val')
        client_test_path = os.path.join(client_path, 'test')
        generate_2014(train_path, (i - 1) / n, i / n, client_train_path)
        generate_2014(val_path, (i - 1) / n, i / n, client_val_path)
        generate_2014(test_path, (i - 1) / n, i / n, client_test_path)
        # 生成标签
        path_list = [client_train_path, client_val_path, client_test_path]
        generate_labels(path_list)
        generate_configs(path_list)


# 生成全局的标注数据
annotations_dir = "/data/projects/dataset/annotations"
train_annotation_file = os.path.join(annotations_dir, 'instances_train2014.json')
val_annotation_file = os.path.join(annotations_dir, 'instances_val2014.json')
test_annotation_file = os.path.join(annotations_dir, 'image_info_test2014.json')

#save_image2labels(train_annotation_file, 'train')
#save_image2labels(val_annotation_file, 'val')
save_image2labels(test_annotation_file, 'test')

# 为每个文件夹生成标注数据以及config.yaml文件
# generate_dataset(8)
