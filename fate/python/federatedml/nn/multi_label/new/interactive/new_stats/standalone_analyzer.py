import matplotlib.pyplot as plt
import pandas as pd

import csv
import os
import statistics


def handle_tensor_mAP(mAPs):
    # mAPs = [float(mAP.strip("tensor()").strip()) / 100 for mAP in mAPs]
    # mAPs = pd.Series(mAPs)
    return mAPs


def draw_loss(paths):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        if path.startswith('resnet'):
            continue
        train_path = os.path.join(path, 'train.csv')
        train_data = pd.read_csv(train_path)

        entropy_loss = train_data['entropy_loss']
        relation_loss = train_data['relation_loss']
        overall_loss = train_data['overall_loss']
        epochs = train_data['epoch']
        plt.plot(epochs, entropy_loss, 'r', label='entropy_loss')
        plt.plot(epochs, relation_loss, 'y', label='relation_loss')
        plt.plot(epochs, overall_loss, 'purple', label='overall_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('train loss')
        # 设置题目
        title = path
        plt.title(f'The losses on {title}')

        # 显示图片
        plt.savefig(f'{path}/losses.svg', dpi=600, format='svg')
        plt.close()


def draw_all(paths):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        print(path)
        train_path = os.path.join(path, 'train.csv')
        valid_path = os.path.join(path, 'valid.csv')
        train_data = pd.read_csv(train_path)
        valid_data = pd.read_csv(valid_path)
        epochs = valid_data['epoch']
        train_mAP = handle_tensor_mAP(train_data['mAP'])
        valid_mAP = handle_tensor_mAP(valid_data['mAP'])

        entropy_loss = train_data['entropy_loss']
        relation_loss = train_data['relation_loss']
        overall_loss = train_data['overall_loss']

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_ylim(0, 100)
        ax1.plot(epochs, train_mAP, 'b', label='train mAP')
        ax1.plot(epochs, valid_mAP, 'g', label='valid mAP')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('mAP')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()

        ax2.plot(epochs, entropy_loss, 'r', label='entropy_loss')
        ax2.plot(epochs, relation_loss, 'y', label='relation_loss')
        ax2.plot(epochs, overall_loss, 'purple', label='overall_loss')
        ax2.legend(loc='upper right')
        ax2.set_ylabel('loss')

        # 设置题目
        title = path
        plt.title(f'The learning curve on {title}')
        # 显示图片
        plt.savefig(f'{path}/{title}.svg', dpi=600, format='svg')
        plt.close()


def draw_train_and_valid(paths):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        train_path = os.path.join(path, 'train.csv')
        valid_path = os.path.join(path, 'valid.csv')
        train_data = pd.read_csv(train_path)
        valid_data = pd.read_csv(valid_path)
        epochs = valid_data['epoch']
        train_mAP = handle_tensor_mAP(train_data['mAP'])
        valid_mAP = handle_tensor_mAP(valid_data['mAP'])

        plt.plot(epochs, train_mAP, 'b', label='train mAP')
        plt.plot(epochs, valid_mAP, 'g', label='valid mAP')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('valid mAP')
        # 设置题目
        title = path
        plt.title(f'The learning curve on {title}')

        # 显示图片
        plt.savefig(f'{path}/{title}.svg', dpi=600, format='svg')
        # plt.show()
        plt.close()


def calculate_stats(float_list):
    minimum = min(float_list)
    maximum = max(float_list)
    mean = statistics.mean(float_list)
    variance = statistics.stdev(float_list)
    return minimum, maximum, mean


paths = ['double-interactive-fixed-relation_16_2_0.0001_0.0001_0.0_0.0_1.0_stats',
         'double-interactive-fixed-relation_16_2_0.0001_0.0001_0.0_0.0_10.0_stats',
         'double-interactive_16_2_0.0001_0.0001_0.0_0.0_1.0_stats',
         'double-interactive_16_2_0.0001_0.0001_0.0_0.0_10.0_stats',
         'double-interactive_32_4_0.0001_0.0001_0.0_0.0_1.0_stats',
         'double-interactive_32_4_0.0001_0.0001_0.0_0.0_5.0_stats',
         'resnet_16_0.0001_stats',
         'single_interactive_16_2_0.0001_0.0001_0.0_0.0_1.0_stats',
         'single_interactive_16_2_0.0001_0.0001_0.0_0.0_10.0_stats',
         'single_interactive_32_4_0.0001_0.0001_0.0_0.0_1.0_stats',
         'single_interactive_32_4_0.0001_0.0001_0.0_0.0_5.0_stats',
         'single_interactive_load_model_32_4_0.0001_0.0001_0.0_0.0_1.0_stats']

draw_train_and_valid(paths)
draw_loss(paths)

for path in paths:
    with open(os.path.join(path, 'valid.csv'), 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        mAP = 0
        for row in reader:
            mAP = max(mAP, float(row.get('mAP')))
        print(f'{path}: {mAP}')

