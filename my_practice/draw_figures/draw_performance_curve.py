import matplotlib.pyplot as plt
from pandas import Series

import csv
import os
import numpy as np


def gen_legends(legends):
    res = []
    for legend in legends:
        if legend == 'fed_avg' or legend == 'fedavg':
            res.append('FedAvg')
        elif legend == 'fpsl':
            res.append('FPSL')
        elif legend == 'flag':
            res.append('FLAG')
        elif legend == 'c_gcn':
            res.append('C-GCN')
        elif legend == 'p_gcn':
            res.append('P-GCN')
        elif legend == 'add_gcn_origin':
            res.append('ADD-GCN')
        elif legend == 'fixed_connect_prob_gcn':
            res.append('FML-DGCN w/o standardization')
        elif legend == 'connect_prob_standard_gcn':
            res.append('FML-DGCN w/o self-connectivity')
        elif legend == 'fixed_prob_standard_gcn':
            res.append('FML-DGCN w/o bridge module')
        elif legend == 'fixed_connect_standard_gcn':
            res.append('FML-DGCN w/o dynamic loss')
        else:
            res.append('FML-DGCN')
    return res


datasets = ['voc2007', 'voc2012', 'coco', 'coco2017']
for dataset in datasets:
    base_dir = f'../experiment_res/AAAI/{dataset}'
    type = 'main'
    if type == 'main':
        # 主体实验
        methods = [
            'fed_avg',
            'flag',
            'fpsl',
            'c_gcn',
            'p_gcn',
            'add_gcn_origin',
            'fixed_connect_prob_standard_gcn'
        ] if dataset != 'coco' else [
            'fed_avg',
            'fpsl',
            'c_gcn',
            'p_gcn',
            'add_gcn_origin',
            'fixed_connect_prob_standard_gcn'
        ]
    else:
        methods = [
            'fixed_connect_prob_gcn',
            'connect_prob_standard_gcn',
            'fixed_prob_standard_gcn',
            'fixed_connect_standard_gcn',
            'fixed_connect_prob_standard_gcn'
        ]

    AmAP_list = dict()

    show_epochs = 100000
    largest = 0
    for method in methods:
        path = os.path.join(base_dir, method)
        clients_path = [os.path.join(path, 'guest/10')]
        for i in range(1, 10):
            clients_path.append(os.path.join(path, f'host/{i}'))
        mAP_lists = []
        min_epoch = 100000
        for i in range(len(clients_path)):
            with open(os.path.join(clients_path[i], 'valid.csv'), 'r') as csv_file:
                reader_list = list(csv.DictReader(csv_file))
                min_epoch = min(min_epoch, len(reader_list))
                mAPs = []
                for j in range(len(reader_list)):
                    name = 'map' if dataset.startswith('coco') else 'mAP'
                    cur_mAP = float(reader_list[j].get(name))
                    mAPs.append(cur_mAP)
                mAP_lists.append(mAPs)
        show_epochs = min(show_epochs, min_epoch)
        # 计算10个客户端的AmAP，得到一个列表，然后对其求均值
        mAP_sum = [0 for _ in range(min_epoch)]
        for i in range(min_epoch):
            for j in range(len(clients_path)):
                mAP_sum[i] += mAP_lists[j][i]
            mAP_sum[i] = round(mAP_sum[i] / 10, 1)
            largest = max(largest, mAP_sum[i])
        AmAP_list[f'{method}'] = mAP_sum

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
        if i == len(methods) - 1:
            plt.plot(x_series, Series(AmAP_list[method][0:show_epochs]), color='b')
        else:
            plt.plot(x_series, Series(AmAP_list[method][0:show_epochs]))
    plt.xlabel(x_axis, fontsize=13)
    plt.ylabel('amAP', fontsize=13)
    # plt.ylim(largest - 10,largest)
    plt.xticks(np.arange(1,show_epochs + 1,4))
    plt.legend(gen_legends(methods), fontsize=12)
    # plt.title('The relation between AmAP and total epochs.')

    save_path = os.path.join(f'amap_convergence_res/{type}', f'res_on_{dataset}.eps')
    plt.savefig(save_path, dpi=300, format='eps')
    plt.close()
