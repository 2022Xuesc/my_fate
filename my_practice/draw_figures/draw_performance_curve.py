import matplotlib.pyplot as plt
from pandas import Series

import csv
import os

dataset = 'coco'
base_dir = f'../experiment_res/AAAI/{dataset}'
methods = [
    'fedavg',
    # 'flag',
    'fpsl',
    'c_gcn',
    'p_gcn',
    'fixed_connect_prob_standard_gcn']

# colors = ['g', 'b', 'r', 'palegreen', 'purple', 'gold']

AmAP_list = dict()

show_epochs = 100000
for method in methods:
    path = os.path.join(base_dir, method)
    if not os.path.exists(path):
        continue
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
                cur_mAP = float(reader_list[j].get('map'))
                mAPs.append(cur_mAP)
            mAP_lists.append(mAPs)
    show_epochs = min(show_epochs, min_epoch)
    # 计算10个客户端的AmAP，得到一个列表，然后对其求均值
    mAP_sum = [0 for _ in range(min_epoch)]
    for i in range(min_epoch):
        for j in range(len(clients_path)):
            mAP_sum[i] += mAP_lists[j][i]
        mAP_sum[i] = round(mAP_sum[i] / 10, 1)
    AmAP_list[f'{method}'] = mAP_sum

x_series = Series(range(show_epochs))
x_axis = 'epoch'

for i in range(len(methods)):
    method = methods[i]
    if i == len(methods) - 1:
        plt.plot(x_series, Series(AmAP_list[method][0:show_epochs]), color='b')
    else:
        plt.plot(x_series, Series(AmAP_list[method][0:show_epochs]))
plt.xlabel(x_axis)
plt.ylabel('AmAP')
plt.legend(methods[0:5] + ['ours'])
plt.title('The relation between AmAP and total epochs.')

save_path = os.path.join('convergence_res', f'res_on_{dataset}.svg')
plt.savefig(save_path, dpi=600, format='svg')
