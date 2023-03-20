import matplotlib.pyplot as plt
import os
import pandas as pd

arbiter_path = 'arbiter'
guest_path = 'guest'
host_path = 'host'


def draw_loss(path, file):
    file_path = os.path.join(path, file)
    data = pd.read_csv(file_path)

    iters = data['agg_iter']
    precisions = data['precision']
    recalls = data['recall']
    avg_losses = data['avgloss']

    fig = plt.figure(figsize=(8, 6))

    # 放在右边
    ax1 = fig.add_subplot(111)
    ax1.set_ylim(0, 1)
    ax1.plot(iters, precisions, 'b', label='precision')
    ax1.plot(iters, recalls, 'g', label='recall')
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


def do_draw(path, file):
    file_path = os.path.join(path, file)
    data = pd.read_csv(file_path)

    epochs = data['epoch']
    precisions = data['precision']
    recalls = data['recall']
    losses = data['loss']

    fig = plt.figure(figsize=(8, 6))

    # 放在右边
    ax1 = fig.add_subplot(111)
    ax1.set_ylim(0, 1)
    ax1.plot(epochs, precisions, 'b', label='precision')
    ax1.plot(epochs, recalls, 'g', label='recall')
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
    plt.savefig(f'{path}_{file.split(".")[0]}.svg', dpi=600, format='svg')
    # plt.show()


def draw(path, loss_file=None, train_file=None, valid_file=None):
    # 绘制
    if train_file:
        do_draw(path, train_file)
    if valid_file:
        do_draw(path, valid_file)
    if loss_file:
        draw_loss(path, loss_file)


draw(host_path, train_file='train.csv', valid_file='valid.csv')
draw(guest_path, train_file='train.csv', valid_file='valid.csv')


draw(arbiter_path, loss_file='avgloss.csv')
