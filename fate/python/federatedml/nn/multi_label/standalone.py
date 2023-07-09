# 服务器与客户端的通用逻辑
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as torch_models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from federatedml.nn.backend.multi_label.meta_learning.maml import MAML

import json
import os
import pickle


class AveragePrecisionMeter(object):
    """
    计算每个类（标签）的平均精度
    给定输入为:
    1. N*K的输出张量output：值越大，置信度越高;
    2. N*K的目标张量target：二值向量，0表示负样本，1表示正样本
    3. 可选的N*1权重向量：每个样本的权重
    N是样本个数，K是类别即标签个数
    """

    # Todo: 这里difficult_examples的含义是什么？
    #  可能存在难以识别的目标（模糊、被遮挡、部分消失），往往需要更加复杂的特征进行识别
    #  为了更加有效评估目标检测算法的性能，一般会对这些目标单独处理
    #  标记为difficult的目标物体可能不会作为正样本、也不会作为负样本，而是作为“无效”样本，不会对评价指标产生影响
    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """将计量器的成员变量重置为空"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor，每个样本对应的每个标签的预测概率向量，和为1
            target (Tensor): binary NxK tensor，表示每个样本的真实标签分布
            weight (optional, Tensor): Nx1 tensor，表示每个样本的权重
        """

        # Todo: 进行一些必要的维度转换与检查
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # 确保存储有足够的大小-->对存储进行扩容
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # 存储预测分数scores和目标值targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """ 返回每个类的平均精度
        Return:
            ap (FloatTensor): 1xK tensor，对应标签（类别）k的平均精度
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        non_zero_labels = 0
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
            if targets.sum() != 0:
                non_zero_labels += 1
        # Todo: 在这里判断不为空的标签个数，直接求均值
        return ap.sum() / non_zero_labels

    @staticmethod
    def average_precision(output, target, difficult_examples=False):

        # 对输出概率进行排序
        # Todo: 这里第0维是K吗？跑一遍GCN进行验证
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # 计算prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        # 遍历排序后的下标即可
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            # 更新正标签的个数
            if label == 1:
                pos_count += 1
            # 更新已遍历总标签的个数
            total_count += 1
            if label == 1:
                # 说明召回水平增加，计算precision
                precision_at_i += pos_count / total_count
        # 除以样本的正标签个数对精度进行平均
        # Todo: 一般不需要该判断语句，每个样本总有正标签
        if pos_count != 0:
            precision_at_i /= pos_count
        # 返回该样本的average precision
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1


class COCO(Dataset):
    def __init__(self, images_dir, config_dir, transforms=None, inp_name=None):
        self.images_dir = images_dir
        self.config_dir = config_dir
        self.transforms = transforms
        self.img_list = []
        self.cat2idx = None
        self.get_anno()

        self.num_classes = len(self.cat2idx)
        self.inp = None
        if inp_name is not None:
            inp_file = os.path.join(self.config_dir, inp_name)
            with open(inp_file, 'rb') as f:
                self.inp = pickle.load(f)
            self.inp_name = inp_name

    def get_anno(self):
        list_path = os.path.join(self.images_dir, 'anno.json')
        self.img_list = json.load(open(list_path, 'r'))
        category_path = os.path.join(self.config_dir, 'category.json')
        self.cat2idx = json.load(open(category_path, 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        img, target = self.get(item)
        # 如果self.inp不为空，说明是在GCN的配置环境下
        if self.inp is not None:
            return (img, self.inp), target
        else:  # 否则使用的是常规的网络，直接返回img和target即可
            return img, target

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        # 读取图像数据
        img = Image.open(os.path.join(self.images_dir, filename)).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        # Todo: 这里负标签设置为0，正标签设置为1
        target = np.zeros(self.num_classes, np.float32)
        target[labels] = 1
        return img, target


# train_path = '/data/projects/iid_dataset/client1/train'
# category_dir = '/data/projects/dataset'
train_path = '/home/klaus125/research/fate/my_practice/dataset/coco/data/guest/train'
category_dir = '/home/klaus125/research/fate/my_practice/dataset/coco'

# 载入数据验证一下
train_dataset = COCO(train_path, config_dir=category_dir, transforms=transforms.Compose([
    # 将图像缩放为256*256
    transforms.Resize(256),
    # 随机裁剪出224*224大小的图像用于训练
    transforms.RandomResizedCrop(224),
    # 将图像进行水平翻转
    transforms.RandomHorizontalFlip(),
    # 转换为张量
    transforms.ToTensor(),
    # 对图像进行归一化，以下两个list分别是RGB通道的均值和标准差
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

# 将train_dataset划分support_dataset和query_dataset，比例为1：1
# total_samples = len(train_dataset)
# Todo: 关于此处数据集划分比例的确定
# num_samples = total_samples // 2
# 将train_dataset划分成两个子集
# support_dataset, query_dataset = torch.utils.data.random_split(train_dataset, [num_samples, num_samples])

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=4, num_workers=16,
    drop_last=True, shuffle=True
)

#
# support_loader = torch.utils.data.DataLoader(
#     dataset=support_dataset, batch_size=4, num_workers=16,
#     drop_last=True, shuffle=True
# )
# query_loader = torch.utils.data.DataLoader(
#     dataset=query_dataset, batch_size=4, num_workers=16,
#     drop_last=True, shuffle=True
# )

total_samples = len(train_dataset)

# 计算每个加载器所需的样本数
num_samples = total_samples // 2

# 创建一个索引列表
# indices = list(range(total_samples))

# 随机打乱索引列表
torch.manual_seed(42)  # 可选的随机种子
indices = torch.randperm(total_samples)

# 将索引列表划分为两个子集
support_indices = indices[:num_samples]
query_indices = indices[num_samples:]

# 创建支持集的数据加载器
support_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=torch.utils.data.SubsetRandomSampler(support_indices),
    batch_size=4,
)

# 创建查询集的数据加载器
query_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=torch.utils.data.SubsetRandomSampler(query_indices)
)

device = 'cuda:0'
criterion = torch.nn.BCELoss()

model = torch_models.resnet101(pretrained=True, num_classes=1000)
# 将最后的全连接层替换掉
model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 80))
torch.nn.init.kaiming_normal_(model.fc[0].weight.data)

model = model.to(device)

named_parameters = list(model.named_parameters())
layers_num = len(named_parameters)
select_list = [i + 1 <= layers_num / 2 for i in range(layers_num)]

select_list_1 = [i <= 155 for i in range(layers_num)]
for name, param in model.named_parameters():
    print(f'name = {name}')
    print(f'param = {param}')

INNER_LR = 0.0001
# 将原模型封装起来
maml = MAML(model, lr=INNER_LR)

# 克隆模型，在训练数据集上进行训练
clone = maml.clone()
sigmoid_func = torch.nn.Sigmoid()

# 1. 对整个train_loader划分support set和query set
ap_meter = AveragePrecisionMeter()
for epoch in range(100):
    # 在support set上进行训练
    for support_step, (inputs, target) in enumerate(support_loader):
        inputs = inputs.to(device)
        target = target.to(device)

        # 计算clone模型在support set的输出
        support_output = clone(inputs)
        # 计算损失
        support_loss = criterion(sigmoid_func(support_output), target)
        # 计算梯度并更新权重，这里会保存中间模型，但并不会直接更新原来的模型
        clone.adapt(support_loss)
        print(f'step={support_step}, loss={support_loss.item()}')

    for query_step, (inputs, target) in enumerate(query_loader):
        inputs = inputs.to(device)
        target = target.to(device)
        # 计算元模型在query set上的梯度并更新
        query_output = clone(inputs)
        query_loss = criterion(sigmoid_func(query_output), target)

        model.zero_grad()
        clone.zero_grad()
        # 让克隆模型保存分类器的梯度，便于进行手动的梯度下降，这样的话分类器的参数进行了两轮更新
        clone.save_classifier_grad()
        # 进行反向传播，model也会具有梯度
        query_loss.backward()
        # Todo: 进行梯度下降，更新分类器的参数
        clone.adapt(query_loss, do_calc=False)
        # 使用Adam优化器进行更新
        # Todo: 封装特征提取器层的参数
        #  特征提取器使用meta_learning学到的参数，而分类器使用clone模型的梯度
        torch.optim.Adam(params=list(model.parameters())[:-2], lr=0.0001).step()

        model.fc.load_state_dict(clone.module.fc.state_dict())
    # 在验证集上统计模型的的mAP指标
    model.eval()
    ap_meter.reset()
    with torch.no_grad():
        for validate_step, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(sigmoid_func(output), target)
            ap_meter.add(output.data, target.data)
    mAP = 100 * ap_meter.value()
    print(f'epoch = {epoch}, mAP = {mAP}')

    # if epoch == 1:
    #     torch.save(model.state_dict(), "model.pth")

# 原论文中的实现方式
# 在support set上进行训练
# for support_step, (inputs, target) in enumerate(support_loader):
#     inputs = inputs.to(device)
#     target = target.to(device)
#     index = len(inputs) // 4
#     support_inputs = inputs[:index]
#     support_target = target[:index]
#
#     query_inputs = inputs[index:]
#     query_target = target[index:]
#
#     # 计算clone模型在support set的输出
#     support_output = clone(support_inputs)
#     # 计算损失
#     support_loss = criterion(sigmoid_func(support_output), support_target)
#     # 计算梯度并更新权重
#     clone.adapt(support_loss)
#
#     # 计算元模型在query set上的梯度并更新
#     query_output = clone(query_inputs)
#     query_loss = criterion(sigmoid_func(query_output), query_target)
#
#     model.zero_grad()
#     clone.zero_grad()
#     clone.save_classifier_grad()
#     # 进行反向传播，model也会具有梯度
#     query_loss.backward()
#     # Todo: 封装最后分类层的参数
#     clone.adapt(query_loss, do_calc=False)
#
#     # 使用Adam优化器进行更新
#     # Todo: 封装特征提取器层的参数
#     #  特征提取器使用meta_learning学到的参数，而分类器使用
#     torch.optim.Adam(params=list(model.parameters())[:-2], lr=0.0001).step()
#
#     model.fc.load_state_dict(clone.module.fc.state_dict())
#     # Todo: 统计精度
