import matplotlib.pyplot as plt
import numpy as np
from pandas import Series

import csv
import os


def gen_legends(legends):
    res = []
    for legend in legends:
        if legend == '.':
            res.append('Entropy')
        else:
            res.append('Mini-batch k-means')
    return res


num_scenes = [5, 10, 15, 20]
for num_scene in num_scenes:
    base_dir = '.'
    methods = ['.', 'kmeans']
    mAP_list = dict()

    show_epochs = 100000
    largest = 0
    for method in methods:
        path = os.path.join(base_dir, method, f'stats_num_scenes_{num_scene}')
        mAP_lists = []
        min_epoch = 100000
        with open(os.path.join(path, 'valid.csv'), 'r') as csv_file:
            reader_list = list(csv.DictReader(csv_file))
            min_epoch = min(min_epoch, len(reader_list))
            mAPs = []
            for j in range(len(reader_list)):
                name = 'map'
                cur_mAP = float(reader_list[j].get(name))
                mAPs.append(cur_mAP)
            mAP_lists.append(mAPs)
        show_epochs = min(show_epochs, min_epoch)
        # 计算10个客户端的AmAP，得到一个列表，然后对其求均值
        mAP_list[f'{method}'] = mAP_lists

    x_series = Series(range(1, show_epochs + 1))
    x_axis = 'epoch'

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))

    plt.tick_params(labelsize=12)

    for i in range(len(methods)):
        method = methods[i]
        path = os.path.join(base_dir, method)
        if not os.path.exists(path):
            continue
        # if i == len(methods) - 1:
        #     plt.plot(x_series, Series(mAP_list[method][0][0:show_epochs]), color='b')
        # else:
        plt.plot(x_series, Series(mAP_list[method][0][0:show_epochs]))
    plt.xlabel(x_axis, fontsize=13)
    plt.ylabel('mAP', fontsize=13)
    # plt.ylim(largest - 10,largest)
    plt.xticks(np.arange(1, show_epochs + 1,2))
    plt.legend(gen_legends(methods), fontsize=12)
    # plt.title('The relation between AmAP and total epochs.')

    save_path = os.path.join(f'map_convergence_res', f'res_{num_scene}.svg')
    plt.savefig(save_path, dpi=600, format='svg')
    plt.close()
