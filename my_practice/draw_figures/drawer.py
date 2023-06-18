import json
import os.path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd

sns.set(font_scale=1.5)


def get_labels_cnts(data_dir):
    anno_path = os.path.join(data_dir, 'anno.json')
    anno = json.load(open(anno_path, 'r'))
    # 该文件夹下所有图像的标签值集合，通过访问anno_json得到
    labels = []
    for img_info in anno:
        img_labels = img_info['labels']
        labels.extend(img_labels)
    return labels

def draw_heatmap(save_dir, data):
    sns.set_context({"figure.figsize": (8, 8)})
    sns.heatmap(data=data, vmin=0.0, vmax=2.0, square=True)
    plt.savefig(f'{save_dir}/heatmap.svg', dpi=600, format='svg')


def prob_log(label):
    return prob(label).log()


def prob(label):
    label = label.add(1)
    return label.div(label.sum())


# 计算KL散度
# client_names是客户端的名称
# label_tensors是每个客户端的标签张量，包含每个标签的样本数
def calc_kl_divergence(client_names, label_tensors):
    # 获取客户端的数目
    n = label_tensors.shape[0]
    # Todo: 创建DataFrame时指定数据类型为float，否则是object
    div_frame = pd.DataFrame(columns=client_names, index=client_names, dtype=float)
    for i in range(n):
        # 每个标签向量，转换成prob，再求log
        log_prob_i = prob_log(label_tensors[i])
        for j in range(n):
            # 每个待比较的向量，转为prob，无需求log
            prob_j = prob(label_tensors[j])
            # 计算KL散度，调用kl_div()函数
            div_frame.iloc[i][j] = F.kl_div(log_prob_i, prob_j, reduction='sum').item()
    return div_frame


def draw_hist(target_dir, data_dirs, num_labels=80):
    for data_dir in data_dirs:
        labels_cnts = get_labels_cnts(data_dir)
        info = data_dir.split('/')
        role, phase = info[-2], info[-1]
        plt.figure(figsize=(12, 8))
        plt.hist(
            labels_cnts,
            bins=list(range(num_labels + 1)),
            edgecolor='b',
            histtype='bar',
            alpha=0.5,
        )
        plt.title(f'{role}_{phase}_label_distribution')
        plt.xlabel('label_id')
        plt.ylabel('label_occurrence')

        plt.savefig(f'{target_dir}/{role}_{phase}_distribution.svg', dpi=600, format='svg')
        plt.cla()


def get_labels_feature(labels, num_labels=80):
    labels_vec = [0] * num_labels
    for label_id in labels:
        labels_vec[label_id] += 1
    return labels_vec


def get_sample_size(data_dir):
    anno_path = os.path.join(data_dir, 'anno.json')
    anno = json.load(open(anno_path, 'r'))
    return len(anno)


client_nums = 10
class_nums = 80
# server_path = '/data/projects/my_dataset'
# save_dir = '.'
# server_path = '/data/projects/clustered_dataset'

server_path = 'anno_json_dir'

save_dir = 'clusters_distribution'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

client_names = [f'client{i + 1}' for i in range(client_nums)]

total_labels = []

# Todo: 记录每个客户端的样本数量，画出直方图
samples = []

# 画直方图
for i in range(client_nums):
    client_id = i + 1
    client_train_path = os.path.join(server_path, f'client{client_id}/train')
    client_valid_path = os.path.join(server_path, f'client{client_id}/val')
    # Todo: 作出分布直方图
    draw_hist(save_dir, [client_train_path, client_valid_path])

    labels = get_labels_cnts(client_train_path)

    total_labels.append(get_labels_feature(labels))

    # 只选取训练集大小
    samples.append(get_sample_size(client_train_path))

div_frame = calc_kl_divergence(client_names=client_names, label_tensors=torch.Tensor(total_labels))
draw_heatmap(save_dir, div_frame)

# Todo: 作出关于samples的柱状图
log_samples = np.log10(samples)

plt.bar(client_names,log_samples)

for i, v in enumerate(log_samples):
    plt.text(i, v + 1, str(samples[i]), ha='center')

plt.title('The distribution of the client data')
plt.ylabel('The size of the client dataset ( log10 ) ')

plt.savefig(f'{save_dir}/dataset_distribution.svg', dpi=600, format='svg')
plt.cla()

