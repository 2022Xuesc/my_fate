import os

import matplotlib.pyplot as plt

guest_train_dir = '/home/klaus125/research/dataset/ms-coco/guest/train'
guest_valid_dir = '/home/klaus125/research/dataset/ms-coco/guest/val'
host_train_dir = '/home/klaus125/research/dataset/ms-coco/host/train'
host_valid_dir = '/home/klaus125/research/dataset/ms-coco/host/val'




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





def draw_hist(data_dirs, num_labels=90):
    for data_dir in data_dirs:
        labels_cnts = get_labels_cnts(data_dir)
        info = data_dir.split('/')
        role,phase = info[-2],info[-1]

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


draw_hist([host_train_dir, guest_train_dir,host_valid_dir,guest_valid_dir])
