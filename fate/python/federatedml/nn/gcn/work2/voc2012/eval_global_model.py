import sys

import numpy as np
import torch

import os

sys.path.append('/data/projects/fate/fate/python')
import torchnet.meter as tnt
from collections import OrderedDict
from federatedml.nn.backend.utils.mylogger.mywriter import MyWriter
from federatedml.nn.backend.utils.loader.dataset_loader import DatasetLoader
from federatedml.nn.backend.multi_label.losses.AsymmetricLoss import *
from federatedml.nn.backend.utils.VOC_APMeter import AveragePrecisionMeter

from federatedml.nn.backend.gcn.models import *

jobid_map = {
    'add_gcn': '202407210221332049530',
    'connect_add_gcn': '202407210448151205460',
    'connect_add_prob_gcn': '202407210703164492330',
    'connect_prob_residual_gcn': '202407210915444664720',
    'connect_prob_redisual_fix_static_gcn': '202407211217440832620',
    # 'add_standard_residual_fix_static_gcn': '202407211429337961150'
}
model_map = {
    'add_gcn': aaai_add_gcn,
    'connect_add_gcn': aaai_connect_add_gcn,
    'connect_add_prob_gcn': aaai_connect_add_prob_gcn,
    'connect_prob_residual_gcn': aaai_connect_prob_residual_gcn,
    'connect_prob_redisual_fix_static_gcn': aaai_connect_prob_residual_fix_static_gcn,
    # 'add_standard_residual_fix_static_gcn': aaai_add_standard_gcn
}
config_map = {
    'add_gcn': {
        "in_channels": 1024
    },
    'connect_add_gcn': {
        "in_channels": 300
    },
    'connect_add_prob_gcn': {
        "in_channels": 300
    },
    'connect_prob_residual_gcn': {
        "in_channels": 300
    },
    'connect_prob_redisual_fix_static_gcn': {
        "in_channels": 300
    },
    'add_standard_residual_fix_static_gcn': {
        "in_channels": 1024
    }
}
# Todo: 创建一个模型骨架，然后替换其参数

dir_prefix = "/data/projects/fate/fateflow/jobs"
pretrained = False
device = 'cuda:2'
num_labels = 20
adjList = np.ndarray((20, 20))

batch_size = 8

dataset = 'voc2012'
category_dir = f'/data/projects/fate/my_practice/dataset/{dataset}'
inp_name = 'voc2012_glove_word2vec.pkl'

# Todo: 全局验证集的目录
valid_path = '/data/projects/dataset/clustered_voc2012/global_val'

ap_meter = AveragePrecisionMeter(difficult_examples=True)
criterion = AsymmetricLossOptimized().to(device)

cur_dir_name = os.getcwd()
my_writer = MyWriter(dir_name=cur_dir_name)

for task_name in jobid_map:
    jobid = jobid_map[task_name]
    if len(jobid) == 0:
        continue
    cur_path = os.path.join(dir_prefix, jobid, f'arbiter/999/gcn_0/{jobid}_gcn_0/0/task_executor')
    cur_path = os.path.join(cur_path, os.listdir(cur_path)[0])
    # Todo: 记录数据的信息

    valid_header = ['epoch', 'mAP', 'loss']
    valid_writer = my_writer.get(f"{task_name}_valid.csv", header=valid_header)
    valid_aps_writer = my_writer.get(f"{task_name}_valid_aps.csv")

    files = sorted(os.listdir(cur_path))
    cnt = 0
    file_prefix = 'global_model'
    for file in files:
        if file.startswith(file_prefix):
            cnt += 1
    in_channel = config_map[task_name]["in_channels"]
    if task_name == 'add_standard_residual_fix_static_gcn':
        model = model_map[task_name](pretrained, adjList, device, num_labels, in_channel, needOptimize=False)
    else:
        model = model_map[task_name](pretrained, adjList, device, num_labels, in_channel)
    print("------------------------")
    print(f"enter task: {task_name}")
    for i in range(cnt):
        file_name = f'{file_prefix}_{i}.npy'
        global_model = np.load(os.path.join(cur_path, file_name), allow_pickle=True)
        agg_tensors = []
        for arr in global_model:
            agg_tensors.append(torch.from_numpy(arr).to(device))
        # Todo: 验证一下是否匹配
        agg_len = len(agg_tensors)
        print(f"dump数据的维度: {agg_len}")
        model_len = len(list(model.parameters()))
        print(f"模型的维度: {model_len}")
        if agg_len != model_len:
            print("不匹配")
            continue
        for param, agg_tensor in zip(model.parameters(), agg_tensors):
            param.data.copy_(agg_tensor)

        bn_data = np.load(os.path.join(cur_path, f'bn_data_{i}.npy'), allow_pickle=True)
        bn_tensors = []
        for arr in bn_data:
            bn_tensors.append(torch.from_numpy(arr).to(device))
        # Todo: 加载bn_data
        idx = 0
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.running_mean.data.copy_(bn_tensors[idx])
                idx += 1
                layer.running_var.data.copy_(bn_tensors[idx])
                idx += 1
        # Todo: 模型加载完毕，开始进行训练
        print(f"{task_name}的模型 {i} 加载成功")
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
                print("progress, validate_step: ", validate_step)
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