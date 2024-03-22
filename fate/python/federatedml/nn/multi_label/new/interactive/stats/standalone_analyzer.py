import matplotlib.pyplot as plt

import csv
import os
import pandas as pd

import statistics


def handle_tensor_mAP(mAPs):
    # mAPs = [float(mAP.strip("tensor()").strip()) / 100 for mAP in mAPs]
    # mAPs = pd.Series(mAPs)
    return mAPs


def draw_train_and_valid(paths):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        # train_path = os.path.join(path, 'train.csv')
        valid_path = os.path.join(path, 'valid.csv')
        # train_data = pd.read_csv(train_path)
        valid_data = pd.read_csv(valid_path)
        epochs = valid_data['epoch']
        # train_mAP = handle_tensor_mAP(train_data['mAP'])
        valid_mAP = handle_tensor_mAP(valid_data['mAP'])

        # plt.plot(epochs, train_mAP, 'b', label='train mAP')
        plt.plot(epochs, valid_mAP, 'g', label='valid mAP')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('valid mAP')
        # 设置题目
        title = path
        plt.title(f'The learning curve on {title}')

        # 显示图片
        plt.savefig(f'{title}.svg', dpi=600, format='svg')
        # plt.show()
        plt.close()

def calculate_stats(float_list):
    minimum = min(float_list)
    maximum = max(float_list)
    mean = statistics.mean(float_list)
    variance = statistics.stdev(float_list)
    return minimum, maximum, mean


paths = ['double-interactive-fixed-relation_16_2_0.0001_0.0001_0.0_0.0_1.0_stats',
         'double-interactive-load-model_16_2_0.0001_0.0001_0.0_0.0_1.0_stats',
         'double-interactive_16_2_0.0001_0.0001_0.0_0.0_1.0_stats',
         'double-interactive_16_2_0.0001_0.0001_0.2_0.1_1.0_stats',
         'resnet_16_0.0001_stats',
         'single_interactive_16_2_0.0001_0.0001_0.0_0.0_1.0_stats',
         'single_interactive_16_2_0.0001_0.0001_0.2_0.1_1.0_stats',
         'single_interactive_load_model_16_2_0.0001_0.0001_0.0_0.0_1.0_stats']


for path in paths:
    with open(os.path.join(path, 'valid.csv'), 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        mAP = 0
        for row in reader:
            mAP = max(mAP, float(row.get('mAP')))
        print(f'{path}: {mAP}')
# draw_train_and_valid('double-interactive-fixed-relation_16_2_0.0001_0.0001_0.0_0.0_1.0_stats')


