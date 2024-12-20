import matplotlib.pyplot as plt
import numpy as np
from pandas import Series

import csv
import os


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
        elif legend == 'add_gcn':
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


datasets = [
    'voc2007',
    'voc2012',
    'coco',
    'coco2017'
]
for dataset in datasets:
    base_dir = f'{dataset}_stats'
    # type = 'ablations'
    type = 'main'
    if type == 'main':
        # 主体实验
        methods = [
            'fed_avg',
            'flag',
            'fpsl',
            'c_gcn',
            'p_gcn',
            'add_gcn',
            'fixed_connect_prob_standard_gcn'
        ] if dataset != 'coco' else [
            'fed_avg',
            'fpsl',
            'c_gcn',
            'p_gcn',
            'add_gcn',
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

    GmAP_list = dict()

    show_epochs = 100000
    for method in methods:
        csv_file_path = os.path.join(base_dir, f'{method}_valid.csv')

        min_epoch = 100000
        with open(csv_file_path, 'r') as csv_file:
            reader_list = list(csv.DictReader(csv_file))
            min_epoch = min(min_epoch, len(reader_list))
            mAPs = []
            for j in range(len(reader_list)):
                cur_mAP = float(reader_list[j].get('mAP'))
                mAPs.append(cur_mAP)
            GmAP_list[method] = mAPs
        show_epochs = min(show_epochs, min_epoch)

    # show_epochs = 10
    x_series = Series(range(1, show_epochs + 1))
    x_axis = 'aggregation iteration'

    fig, ax = plt.subplots()

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))

    plt.tick_params(labelsize=12)

    for i in range(len(methods)):
        method = methods[i]
        if i == len(methods) - 1:
            plt.plot(x_series, Series(GmAP_list[method][0:show_epochs]), color='b')
        else:
            plt.plot(x_series, Series(GmAP_list[method][0:show_epochs]))
    plt.xlabel(x_axis, fontsize=13)
    plt.ylabel('GmAP', fontsize=13)
    plt.xticks(np.arange(1, show_epochs))
    plt.legend(gen_legends(methods), fontsize=12)
    # plt.title('The relation between GmAP and total epochs.')

    save_path = os.path.join(f'gmAP_convergence_res/{type}', f'res_on_{dataset}.svg')
    plt.savefig(save_path, dpi=600, format='svg')
    plt.close()
