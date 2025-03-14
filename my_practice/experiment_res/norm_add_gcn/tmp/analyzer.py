import matplotlib.pyplot as plt
import pandas as pd

import os


def draw_loss(path, file):
    file_path = os.path.join(path, file)
    data = pd.read_csv(file_path)

    iters = data['agg_iter']
    mAPs = data['mAP']
    avg_losses = data['avgloss']

    fig = plt.figure(figsize=(8, 6))

    # 放在右边
    ax1 = fig.add_subplot(111)
    ax1.set_ylim(0, 100)
    ax1.plot(iters, mAPs, 'g', label='mAP')
    ax1.set_xlabel('agg_iter')
    ax1.set_ylabel('rate')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()

    ax2.plot(iters, avg_losses, 'r', label='loss')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('loss')

    # 设置题目
    plt.title('The learning curve on ' + path)
    # 显示图片
    plt.savefig(f'{path}_{file.split(".")[0]}.svg', dpi=600, format='svg')
    plt.close()


def do_draw(path, file):
    file_path = os.path.join(path, file)
    data = pd.read_csv(file_path)
    phase = file.split('.')[0]

    epochs = data['epoch']
    mAPs = data['mAP']
    # 将其中

    if mAPs[0][-1] == ')':
        mAPs = [float(mAP.strip("tensor()").strip()) / 100 for mAP in mAPs]
        mAPs = pd.Series(mAPs)
    losses = data[f'{phase}_loss']

    fig = plt.figure(figsize=(8, 6))

    # 放在右边
    ax1 = fig.add_subplot(111)
    ax1.set_ylim(0, 1)
    ax1.plot(epochs, mAPs, 'b', label='mAP')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('mAP')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()

    ax2.plot(epochs, losses, 'r', label='loss')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('loss')

    # 设置题目
    plt.title('The learning curve on ' + path)
    # 显示图片
    plt.savefig(f'{path}_{file.split(".")[0]}.svg', dpi=600, format='svg')
    # plt.show()
    plt.close()


method_paths = ['add_gcn',
                'add_gcn_dynamic_prob',
                'add_gcn_dynamic_all',
                'gin',
                'norm_gcn_dynamic_all',
                'norm_gin_dynamic_all',
                # 'add_gcn_prev'
                ]


def compare_layer_ratio_method(paths, file):
    is_arbiter = False
    for path in paths:
        if path.startswith('arbiter'):
            file = 'avgloss.csv'
            is_arbiter = True
            x_axis = 'agg_iter'
        else:
            file = 'valid.csv'
            x_axis = 'epoch'

        colors = ['g', 'purple', 'r', 'b', 'orange','black']
        ind = 0
        show_epochs = 8 if is_arbiter else 45
        for method_path in method_paths:
            data = pd.read_csv(os.path.join(method_path, os.path.join(path, file)))
            mAP = data['mAP']
            plt.plot(data[x_axis][:show_epochs], mAP[:show_epochs], colors[ind])
            ind += 1

        plt.xlabel(x_axis)
        plt.ylabel('valid mAP')
        plt.ylim(80, 93)

        plt.legend(method_paths)

        # 设置题目
        plt.title('The relation between mAP and total epochs of ' + path)
        # 显示图片
        # plt.savefig(f'compare/{path}.svg', dpi=600, format='svg')
        if is_arbiter:
            id = 'arbiter'
        else:
            id = path.split('/')[-1]
        dir_name = 'compare_voc'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        save_path = os.path.join(dir_name, f'{id}.svg')
        plt.savefig(save_path, dpi=600, format='svg')
        # plt.show()
        plt.close()


def draw_multiple_loss(path, file):
    file_path = os.path.join(path, file)
    data = pd.read_csv(file_path)

    epochs = data['epoch']
    entropy_loss = data['entropy_loss']
    relation_loss = data['relation_loss']
    overall_loss = data['overall_loss']

    # 放在右边

    plt.plot(epochs, entropy_loss, 'b')
    plt.plot(epochs, relation_loss, 'g')
    plt.plot(epochs, overall_loss, 'r')
    plt.xlabel('epoch')
    plt.ylabel('loss value')

    plt.legend(['entropy_loss', 'relation_loss', 'overall_loss'])

    # 设置题目
    plt.title('The loss curve on ' + path)
    # 显示图片
    plt.savefig(f'{path}_loss.svg', dpi=600, format='svg')
    plt.close()


def draw_losses(paths, file):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        draw_multiple_loss(path, file)


def draw(paths, loss_file=None, train_file=None, valid_file=None):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        # 绘制
        if train_file:
            do_draw(path, train_file)
        if valid_file:
            do_draw(path, valid_file)
        if loss_file:
            draw_loss(path, loss_file)


def handle_tensor_mAP(mAPs):
    # mAPs = [float(mAP.strip("tensor()").strip()) / 100 for mAP in mAPs]
    # mAPs = pd.Series(mAPs)
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
        losses = valid_data[f'loss']

        fig = plt.figure(figsize=(8, 6))

        # 放在右边
        ax1 = fig.add_subplot(111)
        ax1.set_ylim(0, 100)
        ax1.plot(epochs, train_mAP, 'b', label='train mAP')
        ax1.plot(epochs, valid_mAP, 'g', label='valid mAP')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('rate')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()

        ax2.plot(epochs, losses, 'r', label='loss')
        ax2.legend(loc='upper right')
        ax2.set_ylabel('loss')

        # 设置题目
        plt.title('The learning curve on ' + path)
        # 显示图片
        plt.savefig(f'{path}.svg', dpi=600, format='svg')
        # plt.show()
        plt.close()


# draw(host_path, train_file='train.csv', valid_file='valid.csv')
# draw(guest_path, train_file='train.csv', valid_file='valid.csv')
#
#

# Todo: 各个客户端自身的结果分析

# paths = ['agg_kmeans', 'kmeans', 'kmeans', 'c_gcn_with_agg', 'c_gcn_without_agg']
# for path in paths:
#     clients_path = [os.path.join(path, 'guest/10')]
# 
#     for i in range(1, 10):
#         clients_path.append(os.path.join(path, f'host/{i}'))
# 
# 
#     # Todo: 各个客户端的结果分析
#     arbiter_path = os.path.join(path, 'arbiter/999')
#     draw_train_and_valid(clients_path)
#     draw(arbiter_path, loss_file='avgloss.csv')

# Todo: 画一下损失成分
# paths = ['agg_kmeans', 'kmeans', 'kmeans', 'c_gcn_with_agg', 'c_gcn_without_agg']
# for path in paths:
#     clients_path = [os.path.join(path, 'guest/10')]
#
#     for i in range(1, 10):
#         clients_path.append(os.path.join(path, f'host/{i}'))
#     draw_losses(clients_path,'train_loss.csv')


# Todo: 比较方法
clients_path = ['guest/10']

for i in range(1, 10):
    clients_path.append(f'host/{i}')
# 将服务器端也加进去
clients_path.append('arbiter/999')

compare_layer_ratio_method(clients_path, 'valid.csv')
