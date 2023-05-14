import json
import os
import shutil


def save_image2labels(json_path, phase):
    with open(json_path) as f:
        data = json.load(f)

    image2labels = {}

    for annotation in data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        # 将category_id添加到哈希表中
        if image_id not in image2labels.keys():
            image2labels[image_id] = set()
        image2labels[image_id].add(category_id)

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


def get_lstm_image2labels():
    with open('train_lstm_image2labels.json') as f:
        return json.load(f)


# 从图像名称中取出image_id
# COCO_val2014_000000000042

def get_image_id(image_name):
    return int(image_name.split('.')[0].split('_')[-1])


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
            f.write(str(label[i]))
            if i != len(label) - 1:
                f.write(',')
        f.write('\n')


def generate_configs(dir_paths):
    if not isinstance(dir_paths, list):
        dir_paths = [dir_paths]
    for dir_path in dir_paths:
        config_path = os.path.join(dir_path, 'config.yaml')
        if not os.path.exists(config_path):
            file = open(config_path, 'w')
            file.close()


# Todo: 根据标签的出现频率对标签顺序进行排序
def generate_embedding_labels(dir_paths):
    # label_id = 0表示开始，
    # label_id = 91表示结束
    startLabel = 0
    endLabel = 91

    if not isinstance(dir_paths, list):
        dir_paths = [dir_paths]
    for dir_path in dir_paths:
        labels_path = os.path.join(dir_path, 'embedding_labels.txt')
        if os.path.exists(labels_path):
            os.remove(labels_path)

        labels = []
        files = os.listdir(dir_path)
        image2labels = get_lstm_image2labels()
        for filename in files:
            if not filename.startswith("COCO"):
                continue
            # 字典json本地存储后,键改为了str类型
            image_id = str(get_image_id(filename))
            # 有些图片可能未被标注
            if image_id in image2labels.keys():
                label = [filename, startLabel]
                label.extend(image2labels[image_id])
                label.append(endLabel)
                labels.append(label)
        # Todo: 将labels写入文件中
        write_labels(labels, labels_path)


# Todo: 已经废弃
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
        cur = 1
        # 解析phase
        image2labels = get_image2labels(dir_path.split('/')[-1])
        # image2labels = get_image2labels('val')
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
            cur = cur + 1
        # Todo: 将labels写入文件中
        write_labels(labels, labels_path)
        print('Done')


# Todo: 关于COCO数据集的新表示方法
def generate_anno(data, images_dir, phase='val'):
    # 读取对应的image_id数据
    image_id_path = os.path.join(data, '{}_image_id.json'.format(phase))
    image_id = json.load(open(image_id_path, 'r'))
    files = os.listdir(images_dir)
    anno_list = []
    for filename in files:
        # 如果filename不是图像文件名称
        if not filename.startswith('COCO'):
            continue
        cur_img_id = str(get_image_id(filename))
        if cur_img_id not in image_id:
            continue
        anno_list.append(image_id[cur_img_id])
    # 将anno_list存储到图像路径中
    target_file_path = os.path.join(images_dir, 'anno.json')
    json.dump(anno_list, open(target_file_path, 'w'))


def generate_img_id(data, phase):
    anno_dir_path = os.path.join(data, 'annotations')
    annotations_file = json.load(open(os.path.join(anno_dir_path, 'instances_{}2014.json'.format(phase))))
    annotations = annotations_file['annotations']
    # 类别的整体数据对于训练集和测试集来说是一样的
    category = annotations_file['categories']
    # Todo: 建立从标签id到标签名称的映射
    category_id = {}
    for cat in category:
        category_id[cat['id']] = cat['name']
    # Todo: 由于标签id存在中断，因此，对标签名称排序重新生成标签id
    cat2idx = category_to_idx(sorted(category_id.values()))

    # Todo: 建立从img_id到标签的映射
    annotations_id = {}
    for annotation in annotations:
        anno_cat_id = annotation['category_id']
        anno_img_id = annotation['image_id']
        if anno_img_id not in annotations_id:
            annotations_id[anno_img_id] = set()
        annotations_id[anno_img_id].add(cat2idx[category_id[anno_cat_id]])

    # 处理图像信息
    img_id = {}
    images = annotations_file['images']
    for img in images:
        cur_img_id = img['id']
        if cur_img_id not in annotations_id:
            continue
        if cur_img_id not in img_id:
            img_id[cur_img_id] = {}
        img_id[cur_img_id]['file_name'] = img['file_name']
        img_id[cur_img_id]['labels'] = list(annotations_id[cur_img_id])

    # Todo: 将img_id存储起来，便于为特定的图片集生成易于访问的图片+标签集
    image_id_file = os.path.join(data, '{}_image_id.json'.format(phase))
    json.dump(img_id, open(image_id_file, 'w'))

    # 将cat2idx存储起来
    target_file_path = os.path.join(data, 'category.json')
    if not os.path.exists(target_file_path):
        json.dump(cat2idx, open(target_file_path, 'w'))


def category_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


coco_dir = '../../../dataset/coco'

# Todo: 服务器端的未执行代码
# client_nums = 10
# image_dir = "/data/projects/clustered_dataset"
# for i in range(client_nums):
#     client_id = i + 1
    # generate_anno(coco_dir, os.path.join(image_dir, f'client{client_id}/train'), 'train')
    # generate_anno(coco_dir, os.path.join(image_dir, f'client{client_id}/val'), 'val')
    # generate_configs(os.path.join(image_dir, f'client{client_id}/val'))
    # generate_configs(os.path.join(image_dir, f'client{client_id}/train'))

# Todo: 客户端待执行代码
# client_nums = 10
# image_dir = "/home/klaus125/research/dataset/clustered_dataset/"
# for i in range(client_nums):
#     client_id = i + 1
#     generate_anno(coco_dir, os.path.join(image_dir, f'client{client_id}/train'), 'val')

