import os

import matplotlib.pyplot as plt

from my_practice.draw_figures.labels_cnts_getter import get_labels_cnts

local_path = '/home/klaus125/research/dataset/ms-coco'
server_path = '/data/projects/my_dataset'

guest_train_dir = os.path.join(local_path, 'guest/train')
guest_valid_dir = os.path.join(local_path, 'guest/val')
host_train_dir = os.path.join(local_path, 'host_train')
host_valid_dir = os.path.join(local_path, 'host_valid')


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


# draw_hist([host_train_dir, guest_train_dir,host_valid_dir,guest_valid_dir])

client_nums = 8
for i in range(1, client_nums + 1):
    client_train_path = os.path.join(server_path, f'client{i}/train')
    client_valid_path = os.path.join(server_path, f'client{i}/val')
    draw_hist([client_train_path, client_valid_path])
