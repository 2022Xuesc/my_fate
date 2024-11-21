import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import json
import os


def get_sample_size(data_dir):
    anno_path = os.path.join(data_dir, 'anno.json')
    anno = json.load(open(anno_path, 'r'))
    return len(anno)


client_nums = 10
mpl.rcParams['font.family'] = 'SimHei'

dataset = "VOC 2012"
# client_path = '/home/klaus125/research/fate/my_practice/dataset/coco/data/guest/train'

datasets = ['VOC 2007', 'VOC 2012', 'COCO 2014', 'COCO 2017']
for dataset in datasets:
    if dataset == 'COCO 2014':
        server_path = "/data/projects/dataset/clustered_dataset"
    elif dataset == 'VOC 2007':
        server_path = '/data/projects/dataset/voc2007/clustered_voc_expanded'
    elif dataset == 'VOC 2012':
        server_path = '/data/projects/dataset/clustered_voc2012'
    else:
        server_path = '/data/projects/dataset/coco2017/clustered_dataset'
    save_dir = 'new_distribution'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    client_names = [f'c{i + 1}' for i in range(client_nums)]

    total_labels = []

    # Todo: 记录每个客户端的样本数量，画出直方图
    samples = []

    # 画直方图
    for i in range(client_nums):
        client_id = i + 1
        client_train_path = os.path.join(server_path, f'client{client_id}/train')

        # 只选取训练集大小
        samples.append(get_sample_size(client_train_path))

    print(samples)
    # Todo: 作出关于samples的柱状图
    log_samples = np.log10(samples)

    plt.bar(client_names, log_samples)

    for i, v in enumerate(log_samples):
        plt.text(i, v + 0.03, str(samples[i]), ha='center')

    # plt.title(f'Distribution of {dataset} datasets among clients')
    plt.ylabel('客户端数据集大小（log10）')
    
    plt.savefig(f'{save_dir}/{dataset}_dataset_distribution.svg', dpi=600, format='svg')
    plt.close()

