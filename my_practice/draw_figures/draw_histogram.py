import os

import matplotlib.pyplot as plt

guest_train_dir = '/home/klaus125/research/dataset/ms-coco/guest/train'
host_train_dir = '/home/klaus125/research/dataset/ms-coco/host/train'




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
        name = data_dir.split('/')[-2]

        plt.hist(
            labels_cnts,
            bins=list(range(num_labels + 1)),
            edgecolor='b',
            histtype='bar',
            alpha=0.5,
        )
        plt.title(name + '_label_distribution')
        plt.xlabel('label_id')
        plt.ylabel('label_occurrence')

        plt.savefig(name + '_distribution.svg', dpi=600, format='svg')
        plt.cla()


draw_hist([host_train_dir, guest_train_dir])
