import matplotlib.pyplot as plt
from pandas import Series

import csv
import os


def gen_legends(legends):
    res = []
    for legend in legends:
        if legend == '.':
            res.append('entropy')
        else:
            res.append('mini-batch k-means')
    return res



base_dir = '.'
methods = ['.', 'kmeans']
num_scenes = [5, 10, 15, 20]

mAP_list = dict()

show_epochs = 100000
largest = 0
for num_scene in num_scenes:
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
        mAP_list[f'{method}_{num_scene}'] = mAP_lists

x_series = Series(range(show_epochs))
x_axis = 'epoch'

for num_scene in num_scenes:
    for i in range(len(methods)):
        method = methods[i]
        key = f'{method}_{num_scene}'
        path = os.path.join(base_dir, method)
        if not os.path.exists(path):
            continue
        # if i == len(methods) - 1:
        #     plt.plot(x_series, Series(mAP_list[key][0][0:show_epochs]), color='b')
        # else:
        plt.plot(x_series, Series(mAP_list[key][0][0:show_epochs]))
plt.xlabel(x_axis)
plt.ylabel('mAP')
plt.ylim(79,81)
plt.legend(gen_legends(methods))
# plt.title('The relation between AmAP and total epochs.')

save_path = os.path.join(f'map_convergence_res', f'res_overall.svg')
plt.savefig(save_path, dpi=600, format='svg')
plt.close()
