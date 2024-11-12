import matplotlib.pyplot as plt
from pandas import Series

import csv
import os


def gen_legends(legends):
    res = []
    for legend in legends:
        if legend == 'fed_avg':
            res.append('FedAvg')
        elif legend == 'fpsl':
            res.append('FPSL')
        elif legend == 'flag':
            res.append('FLAG')
        elif legend == 'c_gcn':
            res.append('C-GCN')
        elif legend == 'p_gcn':
            res.append('P-GCN')
        else:
            res.append('Ours')
    return res


dataset = 'voc2012'
base_dir = f'{dataset}_stats'
methods = [
    'fed_avg',
    'flag',
    'fpsl',
    'c_gcn',
    'p_gcn',
    'fixed_connect_prob_standard_gcn'
]

# colors = ['g', 'b', 'r', 'palegreen', 'purple', 'gold']

GmAP_list = dict()

show_epochs = 100000
for method in methods:
    csv_file_path = os.path.join(base_dir, f'{method}_valid.csv')
    if not os.path.exists(csv_file_path):
        continue
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
x_series = Series(range(show_epochs))
x_axis = 'epoch'

for i in range(len(methods)):
    method = methods[i]
    if i == len(methods) - 1:
        plt.plot(x_series, Series(GmAP_list[method][0:show_epochs]), color='b')
    else:
        plt.plot(x_series, Series(GmAP_list[method][0:show_epochs]))
plt.xlabel(x_axis)
plt.ylabel('GmAP')

plt.legend(gen_legends(methods))
# plt.legend(methods[0:5] + ['ours'])
plt.title('The relation between GmAP and total epochs.')

save_path = os.path.join('convergence_res', f'res_on_{dataset}.svg')
plt.savefig(save_path, dpi=600, format='svg')
