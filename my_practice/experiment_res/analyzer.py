import matplotlib.pyplot as plt
import os
import pandas as pd


# def draw_loss(path, file):
#     file_path = os.path.join(path, file)
#     data = pd.read_csv(file_path)
#
#     iters = data['agg_iter']
#     precisions = data['precision']
#     recalls = data['recall']
#     avg_losses = data['avgloss']
#
#     fig = plt.figure(figsize=(8, 6))
#
#     # 放在右边
#     ax1 = fig.add_subplot(111)
#     ax1.set_ylim(0, 1)
#     ax1.plot(iters, precisions, 'b', label='precision')
#     ax1.plot(iters, recalls, 'g', label='recall')
#     ax1.set_xlabel('agg_iter')
#     ax1.set_ylabel('rate')
#     ax1.legend(loc='upper left')
#
#     ax2 = ax1.twinx()
#
#     ax2.plot(iters, avg_losses, 'r', label='loss')
#     ax2.legend(loc='upper right')
#     ax2.set_ylabel('loss')
#
#     # 设置题目
#     plt.title('The learning curve on ' + path)
#     # 显示图片
#     plt.savefig(f'{path}_{file.split(".")[0]}.svg', dpi=600, format='svg')
#     plt.close()
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


def do_draw_p_r(path, file):
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
    plt.close()


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
        # mAPs = []
        # for path in paths:
        #     for dir i dirs:
        #         file_path = os.path.join(path, f'{dir}/valid.csv')
        #         mAPs.append(pd.read_csv(file_path))
        file_path1 = os.path.join('sync_fpsl_fixed_ratio_drop', os.path.join(path, file))
        data1 = pd.read_csv(file_path1)

        file_path2 = os.path.join('sync_fpsl_fixed_ratio_save', os.path.join(path, file))
        data2 = pd.read_csv(file_path2)

        baseline_path = os.path.join('sync_fpsl_resnet', os.path.join(path, file))
        data3 = pd.read_csv(baseline_path)
        fpsl_st_mAP = data2['mAP']
        if len(fpsl_st_mAP) == 39:
            epochs = data1[x_axis][:-1]
            fpsl_mAP = data1['mAP'][:-1]
        else:
            epochs = data1[x_axis]
            fpsl_mAP = data1['mAP']
        fpsl_st_mAP = data2['mAP']

        fpsl_baseline_mAP = data3['mAP']
        plt.plot(epochs, fpsl_baseline_mAP, 'g')
        plt.plot(epochs, fpsl_mAP, 'b')
        plt.plot(epochs, fpsl_st_mAP, 'r')
        # plt.ylim(50, max(max(fpsl_st_mAP), max(fpsl_mAP)) + 10)
        plt.ylim(60, 80)
        plt.xlabel(x_axis)
        plt.ylabel('valid mAP')

        # 加竖线
        # cliffs = [12, 20, 24, 32, 36]
        # cliff_heights = fpsl_st_mAP[cliffs]
        # plt.vlines(cliffs, 0, cliff_heights,label='label test', linestyles="dashed", colors='green')
        # for i in range(len(cliffs)):
        #     plt.text(cliffs[i] + 1, 2, cliffs[i], ha='center',color="red")

        plt.legend(['FPSL-Full', 'FPSL-Drop', 'FPSL-Save'])

        # 设置题目
        plt.title('The relation between mAP and total epochs of ' + path)
        # 显示图片
        # plt.savefig(f'compare/{path}.svg', dpi=600, format='svg')
        if is_arbiter:
            id = 'arbiter'
        else:
            id = path.split('/')[-1]
        dir_name = 'compare_layer_ratio'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        save_path = os.path.join(dir_name, f'{id}.svg')
        plt.savefig(save_path, dpi=600, format='svg')
        # plt.show()
        plt.close()


def compare_method(paths, file):
    is_arbiter = False
    for path in paths:
        if path.startswith('arbiter'):
            file = 'avgloss.csv'
            is_arbiter = True
            x_axis = 'agg_iter'
        else:
            file = 'valid.csv'
            x_axis = 'epoch'
        # mAPs = []
        # for path in paths:
        #     for dir i dirs:
        #         file_path = os.path.join(path, f'{dir}/valid.csv')
        #         mAPs.append(pd.read_csv(file_path))
        file_path1 = os.path.join('sync_fpsl_split_half', os.path.join(path, file))
        data1 = pd.read_csv(file_path1)

        file_path2 = os.path.join('sync_fpsl_split_correct', os.path.join(path, file))
        data2 = pd.read_csv(file_path2)

        baseline_path = os.path.join('sync_fpsl_resnet', os.path.join(path, file))
        data3 = pd.read_csv(baseline_path)
        half_mAP = data1['mAP']
        correct_mAP = data2['mAP']
        fpsl_baseline_mAP = data3['mAP']
        plt.plot(data3[x_axis], fpsl_baseline_mAP, 'g')
        plt.plot(data1[x_axis], half_mAP, 'b')
        plt.plot(data2[x_axis], correct_mAP, 'r')
        # plt.ylim(50, max(max(fpsl_st_mAP), max(fpsl_mAP)) + 10)
        plt.ylim(0, 100)
        plt.xlabel(x_axis)
        plt.ylabel('valid mAP')

        # 加竖线
        # cliffs = [12, 20, 24, 32, 36]
        # cliff_heights = half_mAP.values[cliffs]
        # plt.vlines(cliffs, 0, cliff_heights,label='label test', linestyles="dashed", colors='green')
        # for i in range(len(cliffs)):
        #     plt.text(cliffs[i] + 1, 2, cliffs[i], ha='center',color="red")

        plt.legend(['FPSL-Full', 'FPSL-Half', 'FPSL-Correct'])

        # 设置题目
        plt.title('The relation between mAP and total epochs of ' + path)
        # 显示图片
        # plt.savefig(f'compare/{path}.svg', dpi=600, format='svg')
        if is_arbiter:
            id = 'arbiter'
        else:
            id = path.split('/')[-1]
        dir_name = 'compare_split'
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
    reg_loss = data['reg_loss']
    obj_loss = data['obj_loss']
    overall_loss = data['overall_loss']

    # 放在右边

    plt.plot(epochs, obj_loss, 'b')
    plt.plot(epochs, reg_loss, 'g')
    plt.plot(epochs, overall_loss, 'r')
    plt.xlabel('epoch')
    plt.ylabel('loss value')

    plt.legend(['obj_loss', 'reg_loss', 'overall_loss'])

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
        losses = valid_data[f'valid_loss']

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

paths = ['sync_fpsl_fixed_ratio_drop', 'sync_fpsl_fixed_ratio_save', 'sync_fpsl_split_half']
# for path in paths:
#     clients_path = [os.path.join(path, 'guest/10')]
#
#     for i in range(1, 10):
#         clients_path.append(os.path.join(path, f'host/{i}'))

# draw(clients_path, train_file='train.csv', valid_file='valid.csv')
# draw_losses(clients_path, 'loss.csv')

# # Todo: 各个客户端的结果分析
# arbiter_path = os.path.join(path, 'arbiter/999')
# draw_train_and_valid(clients_path)
# draw(arbiter_path, loss_file='avgloss.csv')


# Todo: 比较方法
clients_path = ['guest/10']

for i in range(1, 10):
    clients_path.append(f'host/{i}')
# 将服务器端也加进去
clients_path.append('arbiter/999')
compare_method(clients_path, 'valid.csv')
