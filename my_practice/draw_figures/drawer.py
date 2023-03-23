import os.path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
from collections import Counter

sns.set(font_scale=1.5)


def get_labels_cnts(data_dir):
    labels_path = os.path.join(data_dir, 'labels.txt')
    fp = open(labels_path, 'r')
    labels = []
    for line in fp:
        line.strip('\n')
        info = line.split(',')
        for index in range(1, len(info)):
            if info[index] == '1':
                labels.append(index - 1)
    return labels


def learn_heatmap():
    data = sns.load_dataset('flights') \
        .pivot('year', 'month', 'passengers')
    # print(type(data))
    sns.set_context({"figure.figsize": (8, 8)})
    sns.heatmap(data=data, square=True)
    plt.show()


def draw_heatmap(data):
    sns.set_context({"figure.figsize": (8, 8)})
    sns.heatmap(data=data, square=True)
    plt.savefig('heatmap.svg', dpi=600, format='svg')


def prob_log(label):
    return prob(label).log()


def prob(label):
    label = label.add(1)
    return label.div(label.sum())


def calc_kl_divergence(client_names, label_tensors):
    n = label_tensors.shape[0]
    # Todo: 创建DataFrame时指定数据类型为float，否则是object
    div_frame = pd.DataFrame(columns=client_names, index=client_names, dtype=float)
    for i in range(n):
        log_prob_i = prob_log(label_tensors[i])
        for j in range(n):
            prob_j = prob(label_tensors[j])
            div_frame.iloc[i][j] = F.kl_div(log_prob_i, prob_j, reduction='sum').item()
    return div_frame


def draw_hist(data_dirs, num_labels=90):
    for data_dir in data_dirs:
        labels_cnts = get_labels_cnts(data_dir)
        info = data_dir.split('/')
        role, phase = info[-2], info[-1]

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

        plt.savefig(f'{role}_{phase}_distribution.svg', dpi=600, format='svg')
        plt.cla()


def get_labels_feature(labels):
    labels_vec = [0] * 90
    for label_id in labels:
        labels_vec[label_id] += 1
    return labels_vec


client_nums = 8
class_nums = 90
server_path = '/data/projects/my_dataset'

client_names = [f'client{i}' for i in range(1, client_nums + 1)]

total_labels = []

# 画直方图
for i in range(1, client_nums + 1):
    client_train_path = os.path.join(server_path, f'client{i}/train')
    client_valid_path = os.path.join(server_path, f'client{i}/val')
    # draw_hist([client_train_path, client_valid_path])

    labels = get_labels_cnts(client_train_path)

    total_labels.append(get_labels_feature(labels))

div_frame = calc_kl_divergence(client_names=client_names, label_tensors=torch.Tensor(total_labels))
draw_heatmap(div_frame)
