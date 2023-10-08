# 服务器与客户端的通用逻辑
import time

import math
import numpy as np
import torch.nn
import torchnet.meter as tnt
import torchvision.transforms as transforms

import copy
import csv
import json
import os
import typing
from collections import OrderedDict
from federatedml.framework.homo.blocks import aggregator, random_padding_cipher
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.nn.backend.multi_label.losses.AsymmetricLoss import *
from federatedml.nn.backend.multi_label.models import *
from federatedml.nn.backend.pytorch.data import COCO
from federatedml.param.multi_label_param import MultiLabelParam
from federatedml.util import LOGGER
from federatedml.util.homo_label_encoder import HomoLabelEncoderArbiter
# 导入OMP算法模块
from federatedml.nn.backend.utils.OMP import LabelOMP
from federatedml.nn.backend.utils.APMeter import AveragePrecisionMeter
from federatedml.nn.backend.multi_label.losses.SmoothLoss import *

stats_dir = os.path.join(os.getcwd(), 'stats')
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)

buf_size = 1
# 定义和实验数据记录相关的对象

train_file = open(os.path.join(stats_dir, 'train.csv'), 'w', buffering=buf_size)
train_writer = csv.writer(train_file)
train_writer.writerow(['epoch', 'mAP', 'train_loss'])

valid_file = open(os.path.join(stats_dir, 'valid.csv'), 'w', buffering=buf_size)
valid_writer = csv.writer(valid_file)
valid_writer.writerow(['epoch', 'mAP', 'valid_loss'])

avgloss_file = open(os.path.join(stats_dir, 'avgloss.csv'), 'w', buffering=buf_size)
avgloss_writer = csv.writer(avgloss_file)
avgloss_writer.writerow(['agg_iter', 'mAP', 'avgloss'])


class _FedBaseContext(object):
    def __init__(self, max_num_aggregation, name):
        self.max_num_aggregation = max_num_aggregation
        self._name = name
        self._aggregation_iteration = 0

    def _suffix(self, group: str = "model"):
        # Todo: 注意这里的后缀
        #  self._name --> "default"
        #  group      --> "model"
        #  iteration  --> `当前聚合轮次`
        return (
            self._name,
            group,
            f"{self._aggregation_iteration}",
        )

    def increase_aggregation_iteration(self):
        self._aggregation_iteration += 1

    @property
    def aggregation_iteration(self):
        return self._aggregation_iteration

    # Todo: 这里暂时没有配置early-stop相关的策略
    def finished(self):
        if self._aggregation_iteration >= self.max_num_aggregation:
            return True
        return False


# 创建客户端的上下文
class FedClientContext(_FedBaseContext):
    def __init__(self, max_num_aggregation, aggregate_every_n_epoch, name="feat"):
        super(FedClientContext, self).__init__(max_num_aggregation=max_num_aggregation, name=name)
        self.transfer_variable = SecureAggregatorTransVar()
        self.aggregator = aggregator.Client(self.transfer_variable.aggregator_trans_var)
        self.random_padding_cipher = random_padding_cipher.Client(
            self.transfer_variable.random_padding_cipher_trans_var
        )
        self.aggregate_every_n_epoch = aggregate_every_n_epoch
        self._params: list = []

        self._should_stop = False
        self.loss_summary = []

    def init(self):
        self.random_padding_cipher.create_cipher()

    def encrypt(self, tensor: torch.Tensor, weight):
        return self.random_padding_cipher.encrypt(
            torch.clone(tensor).detach().mul_(weight)
        ).numpy()

    # 发送模型
    # 这里tensors是模型参数，weight是模型聚合权重
    def send_model(self, tensors, bn_data, relation_matrix, weight):
        tensor_arrs = []
        for tensor in tensors:
            tensor_arr = tensor.data.cpu().numpy()
            tensor_arrs.append(tensor_arr)
        # 不仅发送模型参数，还发送bn层的统计数据
        bn_arrs = []
        for bn_item in bn_data:
            bn_arr = bn_item.data.cpu().numpy()
            bn_arrs.append(bn_arr)

        self.aggregator.send_model(
            (tensor_arrs, bn_arrs, relation_matrix, weight), suffix=self._suffix()
        )

    # 接收模型
    def recv_model(self):
        return self.aggregator.get_aggregated_model(suffix=self._suffix())

    def send_loss(self, mAP, loss, weight):
        self.aggregator.send_model((mAP, loss, weight), suffix=self._suffix(group="loss"))

    def recv_loss(self):
        return self.aggregator.get_aggregated_model(
            suffix=self._suffix(group="convergence")
        )

    # 发送、接收全局模型并更新本地模型
    def do_aggregation(self, bn_data, relation_matrix, weight, device):
        # 发送全局模型
        self.send_model(self._params, bn_data, relation_matrix, weight)
        LOGGER.warn("模型发送完毕")

        recv_elements: typing.List = self.recv_model()
        LOGGER.warn("模型接收完毕")
        global_model, bn_data, relation_matrix = recv_elements
        agg_tensors = []
        for arr in global_model:
            agg_tensors.append(torch.from_numpy(arr).to(device))
        for param, agg_tensor in zip(self._params, agg_tensors):
            # param.grad处理的是哪种情况
            if param.grad is None:
                continue
            param.data.copy_(agg_tensor)

        bn_tensors = []
        for arr in bn_data:
            bn_tensors.append(torch.from_numpy(arr).to(device))
        return bn_tensors, relation_matrix

    def do_convergence_check(self, weight, mAP, loss_value):
        self.loss_summary.append(loss_value)
        self.send_loss(mAP, loss_value, weight)
        return self.recv_loss()

    # 配置聚合参数，将优化器中的参数提取出来
    def configure_aggregation_params(self, optimizer):
        if optimizer is not None:
            self._params = [
                param
                # 不是完全倒序，对于嵌套for循环，先声明的在前面
                for param_group in optimizer.param_groups
                for param in param_group["params"]
            ]
            return
        raise TypeError(f"params and optimizer can't be both none")

    def should_aggregate_on_epoch(self, epoch_index):
        return (epoch_index + 1) % self.aggregate_every_n_epoch == 0

    def should_stop(self):
        return self._should_stop

    def set_converged(self):
        self._should_stop = True


# 创建服务器端的上下文
class FedServerContext(_FedBaseContext):
    # Todo: 这里的name关系到FATE架构的通信，至少执行同一联邦学习任务的服务器端和客户端的名称应一样
    def __init__(self, max_num_aggregation, eps=0.0, name="feat"):
        super(FedServerContext, self).__init__(
            max_num_aggregation=max_num_aggregation, name=name
        )
        self.transfer_variable = SecureAggregatorTransVar()
        self.aggregator = aggregator.Server(self.transfer_variable.aggregator_trans_var)
        self.random_padding_cipher = random_padding_cipher.Server(
            self.transfer_variable.random_padding_cipher_trans_var
        )
        self._eps = eps
        self._loss = math.inf

    def init(self, init_aggregation_iteration=0):
        self.random_padding_cipher.exchange_secret_keys()
        self._aggregation_iteration = init_aggregation_iteration

    # 发送模型
    def send_model(self, aggregated_arrs):
        self.aggregator.send_aggregated_model(aggregated_arrs, suffix=self._suffix())

    # 接收客户端模型
    def recv_model(self):
        return self.aggregator.get_models(suffix=self._suffix())

    # 发送收敛状态
    def send_convergence_status(self, mAP, status):
        self.aggregator.send_aggregated_model(
            (mAP, status), suffix=self._suffix(group="convergence")
        )

    def recv_losses(self):
        return self.aggregator.get_models(suffix=self._suffix(group="loss"))

    def do_convergence_check(self):
        loss_weight_pairs = self.recv_losses()
        total_loss = 0.0
        total_weight = 0.0
        total_mAP = 0.

        for mAP, loss, weight in loss_weight_pairs:
            total_loss += loss * weight
            total_mAP += mAP * weight
            total_weight += weight
        mean_loss = total_loss / total_weight
        mean_mAP = total_mAP / total_weight

        avgloss_writer.writerow([self.aggregation_iteration, mean_mAP, mean_loss])

        is_converged = abs(mean_loss - self._loss) < self._eps
        self._loss = mean_loss
        self.send_convergence_status(mean_mAP, is_converged)

        LOGGER.info(f"convergence check: loss={mean_loss}, is_converged={is_converged}")
        return is_converged, mean_loss


class SyncAggregator(object):
    def __init__(self, context: FedServerContext):
        self.context = context
        self.model = None
        self.bn_data = None
        self.relation_matrix = None

    def fit(self, loss_callback):
        while not self.context.finished():
            # Todo: 这里应该是同步接收的方式
            recv_elements: typing.List[typing.Tuple] = self.context.recv_model()

            cur_iteration = self.context.aggregation_iteration
            LOGGER.warn(f'收到{len(recv_elements)}客户端发送过来的模型')

            tensors = [party_tuple[0] for party_tuple in recv_elements]
            # 还有bn层的统计数据
            bn_tensors = [party_tuple[1] for party_tuple in recv_elements]
            # 提取捕捉到的标签相关性
            relation_matrices = [party_tuple[2] for party_tuple in recv_elements]
            # Todo: 记录各个客户端学习到的标签相关性矩阵
            np.save(f'relation_matrices{cur_iteration}', relation_matrices)

            degrees = [party_tuple[3] for party_tuple in recv_elements]
            self.bn_data = self.aggregate_bn_data(bn_tensors, degrees)
            self.relation_matrix = self.aggregate_relation_matrix(relation_matrices, degrees)
            # 分标签进行聚合
            self.aggregate_by_labels(tensors, degrees)

            self.model = tensors[0]
            LOGGER.warn(f'当前聚合轮次为:{cur_iteration}，聚合完成，准备向客户端分发模型')

            self.context.send_model((tensors[0], self.bn_data, self.relation_matrix))

            LOGGER.warn(f'当前聚合轮次为:{cur_iteration}，模型参数分发成功！')
            # 还需要进行收敛验证，目的是统计平均结果
            self.context.do_convergence_check()
            # 同步方式下，服务器端也需要记录聚合轮次
            self.context.increase_aggregation_iteration()

        # Todo: 除了保存参数之外，还要保存BN层的统计数据mean和var
        if self.context.finished():
            print(os.getcwd())
            np.save('global_model', self.model)
            np.save('bn_data', self.bn_data)

    def aggregate_bn_data(self, bn_tensors, degrees):
        degrees = np.array(degrees)
        degrees_sum = degrees.sum(axis=0)

        client_nums = len(bn_tensors)
        layer_nums = len(bn_tensors[0]) // 2
        bn_data = []
        # 遍历每一层
        for i in range(layer_nums):
            mean_idx = i * 2
            mean_var_dim = len(bn_tensors[0][mean_idx])
            mean = np.zeros(mean_var_dim)
            # 遍历每个客户端
            for idx in range(client_nums):
                # 该层在该客户端上的mean是bn_tensors[id][i * 2],方差是bn_tensors[id][i * 2 + 1]
                client_mean = bn_tensors[idx][mean_idx]
                mean += client_mean * degrees[idx][-1]
            mean /= degrees_sum[-1]
            bn_data.append(mean)
            # 计算完均值之后，开始计算方差
            var_idx = mean_idx + 1
            var = np.zeros(mean_var_dim)
            for idx in range(client_nums):
                client_mean = bn_tensors[idx][mean_idx]
                client_var = bn_tensors[idx][var_idx]
                var += (client_var + client_mean ** 2 - mean ** 2) * degrees[idx][-1]
            var /= degrees_sum[-1]
            bn_data.append(var)
        return bn_data

    def aggregate_relation_matrix(self, relation_matrices, degrees):
        degrees = np.array(degrees)
        degrees_sum = degrees.sum(axis=0)
        client_nums = len(relation_matrices)
        relation_matrix = np.zeros_like(relation_matrices[0])
        for i in range(client_nums):
            relation_matrix += relation_matrices[i] * degrees[i][-1] / degrees_sum[-1]
        return relation_matrix

    def aggregate_by_labels(self, tensors, degrees):
        # degrees是91个元素的列表，前90个元素是最后一层各个类别的聚合权重，而最后一个元素是之前层的聚合权重
        # 先聚合之前的特征层，聚合权重为degrees[i][-1]
        # 将degree转为array
        degrees = np.array(degrees)
        degrees_sum = degrees.sum(axis=0)
        # i表示第i个客户端
        for i in range(len(tensors)):
            for j, tensor in enumerate(tensors[i]):
                # 如果是最后两层
                if j == len(tensors[i]) - 2 or j == len(tensors[i]) - 1:
                    # 对每个列向量进行聚合
                    for k in range(len(tensor)):
                        # 对col_vec进行聚合
                        # 如果客户端都不含对应标签的数据，则使用传统方法进行聚合，使得聚合后的权重非0
                        if degrees_sum[k] == 0:
                            tensor[k] *= degrees[i][-1]
                            tensor[k] /= degrees_sum[-1]
                        else:
                            tensor[k] *= degrees[i][k]
                            tensor[k] /= degrees_sum[k]
                        if i != 0:
                            tensors[0][j][k] += tensor[k]
                else:
                    tensor *= degrees[i][-1]
                    tensor /= degrees_sum[-1]
                    if i != 0:
                        tensors[0][j] += tensor
        # 聚合后的权重即为tensors[0]
        return tensors[0]

    def export_model(self, param):
        pass

    @classmethod
    def load_model(cls, model_obj, meta_obj, param):
        param.restore_from_pb(meta_obj.params)

    @classmethod
    def load_model(cls, model_obj, meta_obj, param):
        pass

    @staticmethod
    def dataset_align():
        LOGGER.info("start label alignment")
        label_mapping = HomoLabelEncoderArbiter().label_alignment()
        LOGGER.info(f"label aligned, mapping: {label_mapping}")


def build_aggregator(param: MultiLabelParam, init_iteration=0):
    # Todo: [WARN]
    # param.max_iter = 100
    context = FedServerContext(
        max_num_aggregation=param.max_iter,
        eps=param.early_stop_eps
    )
    context.init(init_aggregation_iteration=init_iteration)
    # Todo: 这里设置同步或异步的聚合方式
    fed_aggregator = SyncAggregator(context)
    return fed_aggregator


def build_fitter(param: MultiLabelParam, train_data, valid_data):
    # Todo: [WARN]
    # param.batch_size = 2
    # param.max_iter = 100
    # param.device = 'cuda:0'

    epochs = param.aggregate_every_n_epoch * param.max_iter
    context = FedClientContext(
        max_num_aggregation=param.max_iter,
        aggregate_every_n_epoch=param.aggregate_every_n_epoch
    )
    # 与服务器进行握手
    context.init()

    # 对数据集构建代码的修改

    # 使用绝对路径
    category_dir = '/data/projects/dataset'
    # category_dir = '/home/klaus125/research/fate/my_practice/dataset/coco'

    # 这里改成服务器路径

    train_dataset = make_dataset(
        data=train_data,
        transforms=train_transforms(),
        category_dir=category_dir
    )
    valid_dataset = make_dataset(
        data=valid_data,
        transforms=valid_transforms(),
        category_dir=category_dir
    )
    batch_size = param.batch_size
    if batch_size < 0 or len(train_dataset) < batch_size:
        batch_size = len(train_dataset)
    shuffle = True

    drop_last = False
    num_workers = 32

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
        drop_last=drop_last, shuffle=shuffle
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size, num_workers=num_workers,
        drop_last=drop_last, shuffle=shuffle
    )
    fitter = MultiLabelFitter(param, epochs, context=context)
    return fitter, train_loader, valid_loader


def make_dataset(data, category_dir, transforms):
    return COCO(data.path, config_dir=category_dir, transforms=transforms)


class MultiLabelFitter(object):
    def __init__(
            self,
            param,
            epochs,
            label_mapping=None,
            context: FedClientContext = None
    ):
        self.scheduler = ...
        self.param = copy.deepcopy(param)
        self._all_consumed_data_aggregated = True
        self.best_precision = 0
        self.context = context
        self.label_mapping = label_mapping

        # Todo: 原始的ResNet101分类器
        (self.model, self.scheduler, self.optimizer) = _init_learner(self.param, self.param.device)

        # 使用非对称损失
        self.criterion = AsymmetricLossOptimized().to(self.param.device)

        # 添加标签平滑损失
        self.lambda_y = 1

        self.start_epoch, self.end_epoch = 0, epochs

        # 聚合策略的相关参数
        # 1. 按照训练使用的总样本数进行聚合
        self._num_data_consumed = 0
        # Todo: 以下两种参数需要知道确切的标签信息，因此，在训练的批次迭代中进行更新
        #  FLAG论文客户端在聚合时可根据标签列表信息直接计算聚合权重
        #  而PS论文需要将标签出现向量发送给服务器端实现分标签聚合
        # 2. 按照训练使用的标签数进行聚合，对应FLAG论文
        self._num_label_consumed = 0
        # 3. 按照每个标签所包含的样本数进行聚合，维护一个list，对应FLAG论文和Partial Supervised论文
        self._num_per_labels = [0] * self.param.num_labels

        # Todo: 创建ap_meter
        self.ap_meter = AveragePrecisionMeter(difficult_examples=False)

        self.lr_scheduler = None

        self.relation_optimizer = None
        # 维护一个邻接表，存储与每个标签相关的其他标签的相关性以及优化变量
        # 自相关性无需进行优化
        # self.adjList = [dict()] * 80，错误实例，这样初始化出来的每个字典都是一样的
        self.adjList = []
        self.variables = []

    def get_label_mapping(self):
        return self.label_mapping

    # 执行拟合操作
    def fit(self, train_loader, valid_loader):

        # 初始化OneCycleLR学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                                max_lr=self.param.lr,
                                                                epochs=self.end_epoch,
                                                                steps_per_epoch=len(train_loader))

        # Todo: 添加优化参数
        #  获取json文件
        image_id2labels = json.load(open(self.param.json_file, 'r'))
        num_labels = 80
        adjList = np.zeros((num_labels, num_labels))
        nums = np.zeros(num_labels)
        for image_info in image_id2labels:
            labels = image_info['labels']
            for label in labels:
                nums[label] += 1
            n = len(labels)
            for i in range(n):
                for j in range(i + 1, n):
                    x = labels[i]
                    y = labels[j]
                    adjList[x][y] += 1
                    adjList[y][x] += 1
        nums = nums[:, np.newaxis]
        # 遍历每一行
        for i in range(num_labels):
            if nums[i] != 0:
                adjList[i] = adjList[i] / nums[i]
        # 遍历A，将主对角线元素设置为1
        for i in range(num_labels):
            adjList[i][i] = 1

        self.construct_relation_by_matrix(adjList)

        for epoch in range(self.start_epoch, self.end_epoch):
            self.on_fit_epoch_start(epoch, len(train_loader.sampler))
            mAP, loss = self.train_validate(epoch, train_loader, valid_loader, self.scheduler)
            LOGGER.warn(f'epoch={epoch}/{self.end_epoch},mAP={mAP},loss={loss}')
            self.on_fit_epoch_end(epoch, valid_loader, mAP, loss)
            if self.context.should_stop():
                break

    # Todo: 聚合依赖数据的更新
    def on_fit_epoch_start(self, epoch, num_samples):
        if self._all_consumed_data_aggregated:
            self._num_data_consumed = num_samples
            self._all_consumed_data_aggregated = False
        else:
            self._num_data_consumed += num_samples

    def on_fit_epoch_end(self, epoch, valid_loader, valid_mAP, valid_loss):
        mAP = valid_mAP
        loss = valid_loss
        if self.context.should_aggregate_on_epoch(epoch):
            self.aggregate_model(epoch, self._num_data_consumed)
            # 同步模式下，需要发送loss和mAP
            mean_mAP, status = self.context.do_convergence_check(
                self._num_data_consumed, mAP, loss
            )
            self._all_consumed_data_aggregated = True

            # 将相关指标重置为0
            self._num_data_consumed = 0
            self._num_label_consumed = 0
            self._num_per_labels = [0] * self.param.num_labels

            self.context.increase_aggregation_iteration()

        # if (epoch + 1) % 50 == 0:
        #     torch.save(self.model.state_dict(), f'model_{epoch + 1}.path')

    # 执行拟合逻辑的编写
    def train_one_epoch(self, epoch, train_loader, scheduler):
        mAP, loss = self.train(train_loader, self.model, self.criterion, self.optimizer,
                               epoch, self.param.device, scheduler)

        train_writer.writerow([epoch, mAP, loss])
        return mAP, loss

    def validate_one_epoch(self, epoch, valid_loader, scheduler):
        mAP, loss = self.validate(valid_loader, self.model, self.criterion, epoch, self.param.device,
                                  scheduler)
        valid_writer.writerow([epoch, mAP, loss])
        return mAP, loss

    def construct_relation_by_matrix(self, matrix):
        num_labels = 80
        self.adjList = [dict() for _ in range(num_labels)]
        self.variables = []
        # 设定阈值th
        th = 0.5
        # 遍历每个值，初始化adjList
        for i in range(num_labels):
            for j in range(num_labels):
                # 自相关性如何处理？额外进行处理
                if i == j or matrix[i][j] < th:
                    continue
                # 优化变量的索引
                idx = len(self.variables)
                variable = torch.tensor(matrix[i][j]).to(self.param.device)
                variable.requires_grad_()
                self.variables.append(variable)
                # 直接维护对应的优化变量即可
                self.adjList[i][j] = variable

        # 构造相关性的优化器
        # 将需要优化的变量列表传入到优化器中
        self.relation_optimizer = torch.optim.SGD(self.variables, lr=0.1)

    def aggregate_model(self, epoch, weight):
        # 配置参数，将优化器optimizer中的参数写入到list中
        self.context.configure_aggregation_params(self.optimizer)

        # bn_data添加
        bn_data = []
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                bn_data.append(layer.running_mean)
                bn_data.append(layer.running_var)

        # Partial Supervised聚合策略
        weight_list = list(self._num_per_labels)
        weight_list.append(self._num_data_consumed)

        labels_num = 80
        A = np.zeros((labels_num, labels_num))
        # 初始化自相关性
        for i in range(labels_num):
            A[i][i] = 1.0
        for i in range(len(self.adjList)):
            # 这里的t是key
            for label in self.adjList[i]:
                A[i][label] = self.adjList[i][label].item()

        agg_bn_data, relation_matrix = self.context.do_aggregation(weight=weight_list, bn_data=bn_data,
                                                                   relation_matrix=A,
                                                                   device=self.param.device)

        self.construct_relation_by_matrix(relation_matrix)

        idx = 0
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.running_mean.data.copy_(agg_bn_data[idx])
                idx += 1
                layer.running_var.data.copy_(agg_bn_data[idx])
                idx += 1

        # 从relation_matrix中重构出adjList和优化器

    def train_validate(self, epoch, train_loader, valid_loader, scheduler):
        mAP, loss = self.train_one_epoch(epoch, train_loader, scheduler)
        if valid_loader:
            mAP, loss = self.validate_one_epoch(epoch, valid_loader, scheduler)
        if self.scheduler:
            self.scheduler.on_epoch_end(epoch, self.optimizer)
        return mAP, loss

    def train(self, train_loader, model, criterion, optimizer, epoch, device, scheduler):
        # 总样本数量为total_samples
        total_samples = len(train_loader.sampler)
        # batch_size
        batch_size = 1 if total_samples < train_loader.batch_size else train_loader.batch_size
        # 记录一个epoch需要执行多少个batches
        steps_per_epoch = math.ceil(total_samples / batch_size)

        self.ap_meter.reset()
        model.train()

        # 对Loss进行更新
        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

        # 对于每一个batch，使用该批次的数据集进行训练
        for train_step, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(device)
            target = target.to(device)

            # Debug看这里的统计是否准确
            self._num_per_labels += target.t().sum(dim=1).cpu().numpy()

            # 也可在聚合时候统计，这里为明了起见，直接统计
            self._num_label_consumed += target.sum().item()

            output = model(inputs)

            self.ap_meter.add(output.data, target.data)

            # 这里只考虑标签平滑损失
            predicts = torch.sigmoid(output).to(torch.float64)

            predict_similarities = LabelOMP(predicts.detach(), self.adjList)


            label_loss = LabelSmoothLoss(relation_need_grad=True)(predicts.detach(), predict_similarities,
                                                                  self.adjList)
            # 需要先对label_loss进行反向传播，梯度下降更新标签相关性

            # 如果标签平滑损失不为0，才进行优化
            if label_loss != 0:
                self.relation_optimizer.zero_grad()
                label_loss.backward()
                self.relation_optimizer.step()
            # 确保标签相关性在0到1之间

            # 遍历每个优化变量，对其值进行约束，限定在[0,1]之内
            for variable in self.variables:
                variable.data = torch.clamp(variable.data, min=0.0, max=1.0)

            # 总损失 = 交叉熵损失 + 标签相关性损失
            # 优化CNN参数时，need_grad设置为True，表示需要梯度

            loss = criterion(output, target) + \
                   self.lambda_y * LabelSmoothLoss(relation_need_grad=False)(predicts, predict_similarities,
                                                                             self.adjList)

            # 初始化标签相关性，先计算标签平滑损失，对相关性进行梯度下降

            losses[OBJECTIVE_LOSS_KEY].add(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.lr_scheduler.step()

        mAP = 100 * self.ap_meter.value()

        return mAP.item(), losses[OBJECTIVE_LOSS_KEY].mean

    def validate(self, valid_loader, model, criterion, epoch, device, scheduler):
        # 对Loss进行更新
        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

        total_samples = len(valid_loader.sampler)
        batch_size = valid_loader.batch_size

        total_steps = math.ceil(total_samples / batch_size)

        sigmoid_func = torch.nn.Sigmoid()

        model.eval()
        # Todo: 在开始训练之前，重置ap_meter
        self.ap_meter.reset()
        with torch.no_grad():
            for validate_step, (inputs, target) in enumerate(valid_loader):
                inputs = inputs.to(device)
                target = target.to(device)
                output = model(inputs)
                loss = criterion(sigmoid_func(output), target)
                losses[OBJECTIVE_LOSS_KEY].add(loss.item())

                # 将输出和对应的target加入到ap_meter中
                # Todo: 对相关格式的验证
                self.ap_meter.add(output.data, target.data)

        mAP = 100 * self.ap_meter.value()
        return mAP.item(), losses[OBJECTIVE_LOSS_KEY].mean


def _init_learner(param, device='cpu'):
    # Todo: 将通用部分提取出来
    model = create_resnet101_model(param.pretrained, device=device, num_classes=param.num_labels)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=1e-4)
    scheduler = None
    return model, scheduler, optimizer


def train_transforms():
    return transforms.Compose([
        # 将图像缩放为256*256
        transforms.Resize(512),
        # 随机裁剪出224*224大小的图像用于训练
        transforms.RandomResizedCrop(448),
        # 将图像进行水平翻转
        transforms.RandomHorizontalFlip(),
        # 转换为张量
        transforms.ToTensor(),
        # 对图像进行归一化，以下两个list分别是RGB通道的均值和标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def valid_transforms():
    return transforms.Compose([
        transforms.Resize(512),
        # 输入图像是224*224
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
