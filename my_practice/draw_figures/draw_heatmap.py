import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd

sns.set(font_scale=1.5)


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
    plt.show()


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


# learn_heatmap()
client_names = ['guest', 'host']
labels = [[100.0, 1000.0, 0, 0], [100.0, 999.0, 0.0, 20.0]]

div_frame = calc_kl_divergence(client_names=client_names, label_tensors=torch.Tensor(labels))
print(div_frame)
data = sns.load_dataset('flights') \
    .pivot('year', 'month', 'passengers')
draw_heatmap(div_frame)
# learn_heatmap()
