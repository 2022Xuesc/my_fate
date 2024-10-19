import matplotlib.pyplot as plt
import os
import pandas as pd


def handle_tensor_mAP(mAPs):
    mAPs = [float(mAP.strip("tensor()").strip()) / 100 for mAP in mAPs]
    mAPs = pd.Series(mAPs)
    return mAPs


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
        plt.ylabel('mAP')
        # 设置题目
        title = 'Voc-Standalone'
        plt.title(f'The learning curve on {title}')

        # 显示图片
        plt.savefig(f'{title}.svg', dpi=600, format='svg')
        # plt.show()
        plt.close()


base_path = "/home/klaus125/research/fate/state/standalone_res"
plateau_path = os.path.join(base_path, "plateau_stats")
# decay_path = os.path.join(base_path, "decay_stats")
# paths = [plateau_path,decay_path]
draw_train_and_valid(plateau_path)
