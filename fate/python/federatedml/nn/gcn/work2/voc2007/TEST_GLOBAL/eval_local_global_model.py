import sys

import numpy as np
import torch

import os
from federatedml.nn.backend.utils.mylogger.mywriter import MyWriter

sys.path.append('/data/projects/fate/fate/python')

from federatedml.nn.backend.utils.loader.dataset_loader import DatasetLoader

from federatedml.nn.backend.multi_label.losses.AsymmetricLoss import *
from federatedml.nn.backend.utils.VOC_APMeter import AveragePrecisionMeter

from federatedml.nn.backend.gcn.models import *

import torchnet.meter as tnt
from collections import OrderedDict

jobid_map = {
    'test': 'test_id',
    'my_add_gcn': '202405221539385446830',
    'my_add_gcn_fpsl': '202405291429462424860',
    'my_add_standard_gcn': '202405221728529366200',
    'my_add_standard_gcn_agg': '202405261000355831150'
}
model_map = {
    'test': pruned_add_gcn_resnet101,
    'my_add_gcn': pruned_add_gcn_resnet101,
    # 'my_add_gcn_fpsl': pruned_add_gcn_resnet101,
    'my_add_standard_gcn': pruned_add_standard_gcn_resnet101,
    'my_add_standard_gcn_agg': pruned_add_standard_gcn_resnet101
}
# Todo: 创建一个模型骨架，然后替换其参数

dir_prefix = "/data/projects/fate/fateflow/jobs"
pretrained = False
device = 'cuda:0'
num_labels = 20
adjList = np.ndarray((20, 20))

in_channel = 300

batch_size = 8

dataset = 'voc_expanded'
category_dir = f'/home/klaus125/research/fate/my_practice/dataset/{dataset}'
inp_name = 'voc_expanded_glove_word2vec.pkl'
# Todo: 测试集的目录

valid_path = '/home/klaus125/research/fate/my_practice/dataset/voc_expanded/data/guest/train'

ap_meter = AveragePrecisionMeter(difficult_examples=True)
criterion = AsymmetricLossOptimized().to(device)

cur_dir_name = os.getcwd()
my_writer = MyWriter(dir_name=cur_dir_name)

for task_name in jobid_map:
    jobid = jobid_map[task_name]
    # cur_path = os.path.join(dir_prefix, jobid, f'arbiter/999/gcn_0/{jobid}_gcn_0/0/task_executor')
    # cur_path = os.path.join(cur_path, os.listdir(cur_path)[0])
    # Todo: 记录数据的信息

    valid_header = ['epoch', 'mAP', 'loss']
    valid_writer = my_writer.get(f"{task_name}_valid.csv", header=valid_header)
    valid_aps_writer = my_writer.get(f"{task_name}_valid_aps.csv")

    # files = sorted(os.listdir(cur_path))
    # cnt = 0
    # file_prefix = 'global_model'
    # for file in files:
    #     if file.startswith(file_prefix):
    #         cnt += 1
    cnt = 1
    model = model_map[task_name](pretrained, adjList, device, num_labels, in_channel, needOptimize=True,
                                 constraint=False)
    for i in range(cnt):
        # file_name = f'{file_prefix}_{i}.npy'
        global_model = np.load('global_model_0.npy', allow_pickle=True)
        agg_tensors = []
        for arr in global_model:
            agg_tensors.append(torch.from_numpy(arr).to(device))
        for param, agg_tensor in zip(model.named_parameters(), agg_tensors):
            param[1].data.copy_(agg_tensor)
        # Todo: 模型加载完毕，开始进行训练
        bn_data = np.load(os.path.join('bn_data_0.npy'), allow_pickle=True)
        # Todo: 加载bn_data
        idx = 0
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.running_mean.data.copy_(bn_data[idx])
                idx += 1
                layer.running_var.data.copy_(bn_data[idx])
                idx += 1
        dataset_loader = DatasetLoader(category_dir, train_path=valid_path, valid_path=valid_path, inp_name=inp_name)
        _, valid_loader = dataset_loader.get_loaders(batch_size, dataset="VOC", drop_last=False)
        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])
        sigmoid_func = torch.nn.Sigmoid()
        model.eval()
        ap_meter.reset()

        with torch.no_grad():
            for validate_step, ((features, inp), target) in enumerate(valid_loader):
                features = features.to(device)
                inp = inp.to(device)
                prev_target = target.clone()
                target[target == 0] = 1
                target[target == -1] = 0
                target = target.to(device)

                cnn_predicts, gcn_predicts, _ = model(features, inp)
                predicts = (cnn_predicts + gcn_predicts) / 2
                # Todo: 将计算结果添加到ap_meter中
                ap_meter.add(predicts.data, prev_target)

                objective_loss = criterion(sigmoid_func(predicts), target)

                losses[OBJECTIVE_LOSS_KEY].add(objective_loss.item())
        mAP, ap = ap_meter.value()
        mAP *= 100
        loss = losses[OBJECTIVE_LOSS_KEY].mean
        valid_writer.writerow([i, mAP.item(), loss])
        valid_aps_writer.writerow(ap)
