import matplotlib.pyplot as plt
import os
import pandas as pd

path = 'λ_0.00025_8_clients_iid'

clients_path = []

clients_path.append(os.path.join(path, 'guest/2'))
for i in range(3, 10):
    clients_path.append(os.path.join(path, f'host/{i}'))

arbiter_path = os.path.join(path, 'arbiter/1')


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
    phase = file.split('.')[0]

    epochs = data['epoch']
    precisions = data['precision']
    recalls = data['recall']
    losses = data[f'{phase}_loss']

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


def draw_multiple_loss(path, file):
    file_path = os.path.join(path, file)
    data = pd.read_csv(file_path)

    epochs = data['epoch']
    reg_loss = data['reg_loss']
    obj_loss = data['obj_loss']
    overall_loss = data['overall_loss']

    # 放在右边

    plt.plot(epochs, obj_loss, 'b')
    plt.plot(epochs, reg_loss, 'g')
    plt.plot(epochs, overall_loss, 'r')
    plt.xlabel('epoch')
    plt.ylabel('loss value')

    plt.legend(['obj_loss','reg_loss','overall_loss'])

    # 设置题目
    plt.title('The loss curve on ' + path)
    # 显示图片
    plt.savefig(f'{path}_loss.svg', dpi=600, format='svg')
    plt.close()

def draw_losses(paths,file):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        draw_multiple_loss(path,file)

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


# draw(host_path, train_file='train.csv', valid_file='valid.csv')
# draw(guest_path, train_file='train.csv', valid_file='valid.csv')
#
#

# draw(clients_path, train_file='train.csv', valid_file='valid.csv')
#
# draw(arbiter_path, loss_file='avgloss.csv')

draw_losses(clients_path, 'loss.csv')
