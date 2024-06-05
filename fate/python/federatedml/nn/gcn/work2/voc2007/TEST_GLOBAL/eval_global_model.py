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
    # 'my_add_gcn_fpsl': '202405291853581396420',
    # 'my_add_standard_gcn': '202405300648009006500',
    # 'my_add_standard_gcn_fpsl': '202405300824559105990',
    # 'my_add_gin': '202405300402102596790',
    # 'my_add_gin_fpsl': '202405301204117812590',
    # 'my_add_gcn': '202405301427122521060',
    # 'my_connect_add_standard_gcn': '202405301531543889900',
    'my_connect_add_standard_gcn_fpsl': '202405301741117483270',
    'connect_add_gcn_fedavg': '202406041540263877510',
    'connect_add_gcn_fpsl_meta': '202406051021109290830',
    'connect_add_gcn_prob': '202406050247596100450',
    'connect_add_gcn_advanced_prob': '202406050612249408270',
    # 'connect_add_gcn_gap': '202406051531587900350'
}
model_map = {
    # 'my_add_gcn': pruned_add_gcn_resnet101,
    # 'my_add_gcn_fpsl': pruned_add_gcn_resnet101,
    # 'my_add_standard_gcn': pruned_add_standard_gcn_resnet101,
    # 'my_add_standard_gcn_fpsl': pruned_add_standard_gcn_resnet101,
    # 'my_add_gin': pruned_add_gin_resnet101,
    # 'my_add_gin_fpsl': pruned_add_gin_resnet101,
    # 'my_connect_add_standard_gcn': connect_add_standard_gcn,
    'my_connect_add_standard_gcn_fpsl': connect_add_standard_gcn,
    # Todo: 注意参数配置的不同
    'connect_add_gcn_fedavg': connect_add_gcn,
    'connect_add_gcn_fpsl_meta': connect_add_gcn,  # 对于以及训练好的元学习模型，没什么不同
    'connect_add_gcn_prob': connect_add_gcn,
    'connect_add_gcn_advanced_prob': connect_add_gcn,
    'connect_add_gcn_gap': connect_add_gcn
}
config_map = {
    'my_connect_add_standard_gcn_fpsl': [False, False],
    # Todo: 注意参数配置的不同
    'connect_add_gcn_fedavg': [False, False],
    'connect_add_gcn_fpsl_meta': [False, False],
    'connect_add_gcn_prob': [True, False],
    'connect_add_gcn_advanced_prob': [True, True],
    'connect_add_gcn_gap': [False, True]
}
# Todo: 创建一个模型骨架，然后替换其参数

dir_prefix = "/data/projects/fate/fateflow/jobs"
pretrained = False
device = 'cuda:2'
num_labels = 20
adjList = np.ndarray((20, 20))

in_channel = 300

batch_size = 8

dataset = 'voc_expanded'
category_dir = f'/data/projects/fate/my_practice/dataset/{dataset}'
inp_name = 'voc_expanded_glove_word2vec.pkl'
# Todo: 测试集的目录

valid_path = '/data/projects/dataset/voc2007/clustered_voc_expanded/global_val'

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
    prob, gap = config_map[task_name]
    model = model_map[task_name](pretrained, adjList, device, num_labels, in_channel, needOptimize=True,
                                 constraint=False,prob=prob,gap=gap)
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
