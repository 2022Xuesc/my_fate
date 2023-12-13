# 服务器与客户端的通用逻辑
import torch.nn

import json
import os
from federatedml.nn.backend.gcn.models import *
from federatedml.nn.backend.multi_label.losses.AsymmetricLoss import *
from federatedml.nn.backend.utils.APMeter import AveragePrecisionMeter
from federatedml.nn.backend.utils.loader.dataset_loader import DatasetLoader
from federatedml.nn.backend.utils.mylogger.mywriter import MyWriter

# Todo: 场景数量
num_scenes = 20
epochs = 40
batch_size = 8
device = 'cuda:0'


lr, lrp = 0.0001, 0.1
num_classes = 80

stats_dir = f'stats_num_scenes_{num_scenes}'
my_writer = MyWriter(dir_name=os.getcwd(), stats_name=stats_dir)

client_header = ['epoch', 'OP', 'OR', 'OF1', 'CP', 'CR', 'CF1', 'OP_3', 'OR_3', 'OF1_3', 'CP_3', 'CR_3', 'CF1_3', 'map']

train_writer = my_writer.get("train.csv", header=client_header)
valid_writer = my_writer.get("valid.csv", header=client_header)

# 准备模型相关
model = resnet_salgl(pretrained=True, device=device, num_scenes=num_scenes, num_classes=num_classes)
optimizer = torch.optim.AdamW(model.get_config_optim(lr=lr, lrp=lrp), lr=lr, weight_decay=1e-4)

criterion = AsymmetricLossOptimized().to(device)

ap_meter = AveragePrecisionMeter(difficult_examples=False)

# 准备数据集
dataset = "coco"
inp_name = f'{dataset}_glove_word2vec.pkl'

# category_dir = '/home/klaus125/research/fate/my_practice/dataset/coco'
# train_path = '/home/klaus125/research/fate/my_practice/dataset/coco/data/guest/train'
# valid_path = '/home/klaus125/research/fate/my_practice/dataset/coco/data/guest/train'

category_dir = f'/data/projects/fate/my_practice/dataset/{dataset}'
train_path = '/data/projects/dataset/clustered_dataset/client1/train'
valid_path = '/data/projects/dataset/clustered_dataset/client1/val'

dataset_loader = DatasetLoader(category_dir, train_path, valid_path, inp_name)
train_loader, valid_loader = dataset_loader.get_loaders(batch_size)

sigmoid_func = torch.nn.Sigmoid()

# 开始训练
for epoch in range(epochs):
    # Todo: 每个epoch，维护每张图像的场景预测结果，使用dict
    scene_images = dict()
    for i in range(num_scenes):
        scene_images[i] = []
    # 训练阶段
    ap_meter.reset()
    model.train()
    for train_step, ((features, inp), target) in enumerate(train_loader):
        features = features.to(device)
        target = target.to(device)
        inp = inp.to(device)
        output = model(features, inp, y=target)
        predicts = output['output']

        entropy_loss = output['entropy_loss']
        scene_indices = output['scene_indices']
        # Todo: 在这里统计，检索到图像名称并建立关系
        # 这是loader的索引，根据该索引可以索引到图像名称以及标签
        indices = list(train_loader.batch_sampler)[train_step]
        for i in range(len(indices)):
            index = indices[i]
            scene_images[scene_indices[i].item()].append(train_loader.dataset.img_list[index])

        ap_meter.add(predicts.data, target)

        objective_loss = criterion(sigmoid_func(predicts), target)

        overall_loss = objective_loss + entropy_loss

        optimizer.zero_grad()

        overall_loss.backward()

        optimizer.step()
    if (epoch + 1) % 4 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9
    mAP, _ = ap_meter.value()
    mAP *= 100
    # 统计指标
    OP, OR, OF1, CP, CR, CF1 = ap_meter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = ap_meter.overall_topk(3)
    metrics = [OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, mAP.item()]
    train_writer.writerow([epoch] + metrics)

    json_path = f'{stats_dir}/scene_images_{epoch}.json'
    json_content = json.dumps(scene_images, indent=4)
    with open(json_path, 'w') as json_file:
        json_file.write(json_content)

    # 验证阶段
    model.eval()
    ap_meter.reset()
    with torch.no_grad():
        for validate_step, ((features, inp), target) in enumerate(valid_loader):
            features = features.to(device)
            inp = inp.to(device)
            target = target.to(device)

            output = model(features, inp, y=target)
            predicts = output['output']

            # Todo: 将计算结果添加到ap_meter中
            ap_meter.add(predicts.data, target)

            # objective_loss = criterion(sigmoid_func(predicts), target)

    mAP, _ = ap_meter.value()
    mAP *= 100
    OP, OR, OF1, CP, CR, CF1 = ap_meter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = ap_meter.overall_topk(3)
    metrics = [OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, mAP.item()]
    valid_writer.writerow([epoch] + metrics)
