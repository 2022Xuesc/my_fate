import json
import os
import shutil


def save_image2labels(json_path, phase):
    with open(json_path) as f:
        data = json.load(f)

    image2labels = {}

    total_cnt = len(data['annotations'])
    cur = 0
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        # 将category_id添加到哈希表中
        if image_id not in image2labels.keys():
            image2labels[image_id] = set()
        image2labels[image_id].add(category_id)
        cur = cur + 1
        print('progress: {}/{}'.format(cur, total_cnt))


    for image_id in image2labels.keys():
        image2labels[image_id] = list(image2labels[image_id])

    # 将image2labels进行持久化
    image2labels_path = phase + '_image2labels.json'
    json_str = json.dumps(image2labels, indent=4)
    with open(image2labels_path, 'w') as json_file:
        json_file.write(json_str)


def get_image2labels(phase):
    if phase.startswith('train'):
        phase = 'train'
    else:
        phase = 'val'
    with open(phase + '_image2labels.json') as f:
        return json.load(f)


# 从图像名称中取出image_id
# COCO_val2014_000000000042

def get_image_id(image_name):
    # print(image_name)
    return int(image_name.split('.')[0][-12:])


def clear_dir(dir_path):
    if os.path.exists(dir_path):
        files = os.listdir(dir_path)
        for file in files:
            os.remove(os.path.join(dir_path, file))
    else:
        os.makedirs(dir_path)


def generate_2014(src_dir, left, right, target_dir):
    clear_dir(target_dir)
    # 清空target_dir下的所有文件
    files = os.listdir(src_dir)
    cnt = len(files)
    for filename in files[int(left * cnt):int(right * cnt)]:
        fullpath = os.path.join(src_dir, filename)
        shutil.copy(fullpath, target_dir)


# 为每个数据集生成标签

def write_labels(labels, label_path):
    f = open(label_path, 'w')
    for label in labels:
        for i in range(len(label)):
            f.write(label[i])
            if i != len(label) - 1:
                f.write(',')
        f.write('\n')


def generate_configs(dir_paths):
    for dir_path in dir_paths:
        config_path = os.path.join(dir_path, 'config.yaml')
        if not os.path.exists(config_path):
            file = open(config_path, 'w')
            file.close()


def generate_labels(dir_paths):
    if not isinstance(dir_paths, list):
        dir_paths = [dir_paths]
    for dir_path in dir_paths:
        labels_path = os.path.join(dir_path, 'labels.txt')
        if os.path.exists(labels_path):
            os.remove(labels_path)

        labels = []
        files = os.listdir(dir_path)
        files_cnt = len(files)
        cur = 0
        # 解析phase
        image2labels = get_image2labels(dir_path.split('/')[-1])
        for filename in files:
            if filename in ['labels.txt', 'config.yaml']:
                continue
            # 字典json本地存储后,键改为了str类型
            image_id = str(get_image_id(filename))
            # 有些图片可能未被标注
            if image_id in image2labels.keys():
                # label是一个90维度的张量
                label = [filename]
                label.extend(['0'] * 90)
                for id_index in image2labels[image_id]:
                    label[id_index] = '1'
                labels.append(label)
            print(f'progress: {cur}/{files_cnt}')
        # Todo: 将labels写入文件中
        write_labels(labels, labels_path)
        print('Done')


src_path = '/data/projects/dataset/train2014'
generate_labels(src_path)
