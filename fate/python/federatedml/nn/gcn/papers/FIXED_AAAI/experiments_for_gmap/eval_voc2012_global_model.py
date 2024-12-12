import sys

import numpy as np

import os

sys.path.append('/data/projects/fate/fate/python')
import torchnet.meter as tnt
from collections import OrderedDict
from federatedml.nn.backend.utils.mylogger.mywriter import MyWriter
from federatedml.nn.backend.utils.loader.dataset_loader import DatasetLoader
from federatedml.nn.backend.multi_label.losses.AsymmetricLoss import *
from federatedml.nn.backend.utils.VOC_APMeter import AveragePrecisionMeter
from federatedml.nn.backend.multi_label.models import *
from federatedml.nn.backend.gcn.models import *

FED_AVG = 'fed_avg'
FLAG = 'flag'
FPSL = 'fpsl'
C_GCN = 'c_gcn'
P_GCN = 'p_gcn'
OURS = 'fixed_connect_prob_standard_gcn'
WITHOUT_FIX = 'connect_prob_standard_gcn'
WITHOUT_CONNECT = 'fixed_prob_standard_gcn'
ADD_GCN = 'add_gcn'

jobid_map = {
    # FED_AVG: '202410220416373841590',
    # FLAG: '202410220442101237570',
    # FPSL: '202410220819086555390',
    # C_GCN: '202411040306089134600',
    # P_GCN: '202411040308244428370',
    # OURS: '202411070357178860740',
    # WITHOUT_FIX: '202410250218416948480',
    # WITHOUT_CONNECT: '202410250650256294940',
    ADD_GCN: '202412111445209585300'
}
model_map = {
    FED_AVG: create_resnet101_model,
    FLAG: create_resnet101_model,
    FPSL: create_resnet101_model,
    C_GCN: resnet_c_gcn,
    P_GCN: p_gcn_resnet101,
    OURS: aaai_fixed_connect_prob_standard_gcn,
    WITHOUT_FIX: aaai_connect_prob_standard_gcn,
    WITHOUT_CONNECT: aaai_fixed_prob_standard_gcn,
    ADD_GCN: origin_add_gcn
}

# 1个入参，两个返回值：1
# 2个入参，两个返回值：2
# 1个入参，三个返回值：3
# 2个入参，三个返回值：4
# 2个入参，1个返回值 ：5

config_map = {
    FED_AVG: {
        "in_channels": 0,
        "argument_and_return_type": 0
    },
    FPSL: {
        "in_channels": 0,
        "argument_and_return_type": 0
    },
    FLAG: {
        "in_channels": 0,
        "argument_and_return_type": 0
    },
    C_GCN: {
        "in_channels": 300,
        "argument_and_return_type": 5
    },
    P_GCN: {
        "in_channels": 2048,
        "argument_and_return_type": 5
    },
    OURS: {
        "in_channels": 300,
        "argument_and_return_type": 4
    },
    WITHOUT_FIX: {
        "in_channels": 300,
        "argument_and_return_type": 4
    },
    WITHOUT_CONNECT: {
        "in_channels": 300,
        "argument_and_return_type": 4
    },
    ADD_GCN: {
        "in_channels": -1,
        "argument_and_return_type": 1
    }
}
# Todo: 创建一个模型骨架，然后替换其参数

dir_prefix = "/data/projects/fate/fateflow/jobs"
pretrained = False
device = 'cuda:0'
num_labels = 20
adjList = np.ndarray((20, 20))
# Todo: adjList是否优化，会导致不同的结果

batch_size = 8

dataset = 'voc2012'
category_dir = f'/data/projects/fate/my_practice/dataset/{dataset}'
inp_name = 'voc2012_glove_word2vec.pkl'

# Todo: 全局验证集的目录
valid_path = '/data/projects/dataset/clustered_voc2012/global_val'

ap_meter = AveragePrecisionMeter(difficult_examples=True)
criterion = AsymmetricLossOptimized().to(device)

# model = model_map[C_GCN](pretrained, adjList, device, num_labels, 300)
cur_dir_name = os.getcwd()
my_writer = MyWriter(dir_name=cur_dir_name, stats_name='voc2012_stats')

for task_name in jobid_map:
    is_multi_label = task_name.startswith('f') and not task_name.startswith('fixed')
    jobid = jobid_map[task_name]
    if len(jobid) == 0:
        continue
    if not is_multi_label:
        cur_path = os.path.join(dir_prefix, jobid, f'arbiter/999/gcn_0/{jobid}_gcn_0/0/task_executor')
    else:
        cur_path = os.path.join(dir_prefix, jobid, f'arbiter/999/multi_label_0/{jobid}_multi_label_0/0/task_executor')
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
    if is_multi_label or task_name == ADD_GCN:
        model = model_map[task_name](pretrained, device, num_labels)
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

        with_relation = task_name == C_GCN or task_name == P_GCN
        if with_relation:
            if agg_len != model_len - 1:
                print("不匹配")
                continue
        elif agg_len != model_len:
            print("不匹配")
            continue
        lr = 0.1
        lrp = 0.1
        if not is_multi_label:
            optimizer = torch.optim.AdamW(model.get_config_optim(lr=lr, lrp=lrp),
                                          lr=lr,
                                          weight_decay=1e-4)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=1e-4)
        _params = [
            param
            # 不是完全倒序，对于嵌套for循环，先声明的在前面
            for param_group in optimizer.param_groups
            for param in param_group["params"]
        ]

        for param, agg_tensor in zip(_params, agg_tensors):
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
        # Todo: 装载adjList
        adj_path = os.path.join(cur_path, f'relation_matrix_{i}.npy')
        if os.path.exists(adj_path):
            model.updateA(np.load(adj_path, allow_pickle=True))

        # Todo: 模型加载完毕，开始进行训练
        print(f"{task_name}的模型 {i} 加载成功")
        dataset_loader = DatasetLoader(category_dir, train_path=valid_path, valid_path=valid_path, inp_name=inp_name)
        _, valid_loader = dataset_loader.get_loaders(batch_size, dataset="VOC", drop_last=True)
        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])
        sigmoid_func = torch.nn.Sigmoid()
        model.eval()
        ap_meter.reset()

        with torch.no_grad():
            if not is_multi_label:
                for validate_step, ((features, inp), target) in enumerate(valid_loader):
                    print("progress, validate_step: ", validate_step)
                    features = features.to(device)
                    inp = inp.to(device)
                    prev_target = target.clone()
                    target[target == 0] = 1
                    target[target == -1] = 0
                    target = target.to(device)

                    type = config_map[task_name]["argument_and_return_type"]
                    if type == 1:
                        cnn_predicts, gcn_predicts = model(features)
                    elif type == 2:
                        cnn_predicts, gcn_predicts = model(features, inp)
                    elif type == 3:
                        cnn_predicts, gcn_predicts, dynamic_adj_loss = model(features)
                    elif type == 4:
                        cnn_predicts, gcn_predicts, dynamic_adj_loss = model(features, inp)
                    else:
                        gcn_predicts = model(features, inp)
                    predicts = gcn_predicts if type == 5 else (cnn_predicts + gcn_predicts) / 2
                    # Todo: 将计算结果添加到ap_meter中
                    ap_meter.add(predicts.data, prev_target)

                    objective_loss = criterion(sigmoid_func(predicts), target)

                    losses[OBJECTIVE_LOSS_KEY].add(objective_loss.item())
            else:
                for validate_step, ((inputs, inp), target) in enumerate(valid_loader):
                    print("progress, validate_step: ", validate_step)
                    inputs = inputs.to(device)
                    prev_target = target.clone()
                    target[target == 0] = 1
                    target[target == -1] = 0
                    target = target.to(device)

                    output = model(inputs)
                    loss = criterion(sigmoid_func(output), target)
                    losses[OBJECTIVE_LOSS_KEY].add(loss.item())
                    ap_meter.add(output.data, prev_target)
        mAP, ap = ap_meter.value()
        mAP *= 100
        loss = losses[OBJECTIVE_LOSS_KEY].mean
        valid_writer.writerow([i, mAP.item(), loss])
        valid_aps_writer.writerow(ap)
