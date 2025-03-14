# 服务器与客户端的通用逻辑
import copy
import json
from collections import OrderedDict

import numpy as np
import typing
import torchvision.transforms as transforms
import torchnet.meter as tnt
import math
import os
import logging
import csv

import torch
import torch.nn

from federatedml.nn.backend.gcn.config import config_scheduler
from federatedml.nn.backend.gcn.models import *
from federatedml.nn.backend.pytorch.data import COCO
from federatedml.nn.homo_nn import _consts
from federatedml.param.gcn_param import GCNParam
from federatedml.util import LOGGER
from federatedml.framework.homo.blocks import aggregator, random_padding_cipher
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.util.homo_label_encoder import HomoLabelEncoderArbiter
from federatedml.nn.backend.multi_label.losses.AsymmetricLoss import *
from federatedml.nn.backend.utils.APMeter import AveragePrecisionMeter
from federatedml.nn.backend.gcn.utils import MultiScaleCrop, Warp
from federatedml.nn.backend.utils.mylogger.mywriter import MyWriter
from federatedml.nn.backend.utils.aggregators.aggregator import *

from federatedml.nn.backend.utils.loader.dataset_loader import DatasetLoader

my_writer = MyWriter(dir_name=os.getcwd())

client_header = ['epoch', 'OP', 'OR', 'OF1', 'CP', 'CR', 'CF1', 'OP_3', 'OR_3', 'OF1_3', 'CP_3', 'CR_3', 'CF1_3', 'map',
                 'loss']
server_header = ['agg_iter', 'OP', 'OR', 'OF1', 'CP', 'CR', 'CF1', 'OP_3', 'OR_3', 'OF1_3', 'CP_3', 'CR_3', 'CF1_3',
                 'map', 'loss']
train_writer = my_writer.get("train.csv", header=client_header)
valid_writer = my_writer.get("valid.csv", header=client_header)
avgloss_writer = my_writer.get("avgloss.csv", header=server_header)
# 训练时的损失组成记录
train_loss_writer = my_writer.get("train_loss.csv", header=['epoch', 'objective_loss', 'entropy_loss', 'overall_loss'])


class _FedBaseContext(object):
    def __init__(self, max_num_aggregation, name):
        self._name = name

        # Todo: 客户端设置最大聚合轮次和当前聚合轮次
        #  供同步和简单异步使用
        self.max_num_aggregation = max_num_aggregation
        self._aggregation_iteration = 0

    # Todo: 定义发送消息的后缀
    #  会变化的是当前聚合轮次，表示为哪个回合的模型发送聚合权重
    def _suffix(self, group: str = "model"):
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

    def finished(self):
        if self._aggregation_iteration >= self.max_num_aggregation:
            return True
        return False


# 创建客户端的上下文
class FedClientContext(_FedBaseContext):
    def __init__(self, max_num_aggregation, aggregate_every_n_epoch, name="default"):
        super(FedClientContext, self).__init__(max_num_aggregation=max_num_aggregation, name=name)
        self.transfer_variable = SecureAggregatorTransVar()
        self.aggregator = aggregator.Client(self.transfer_variable.aggregator_trans_var)
        self.random_padding_cipher = random_padding_cipher.Client(
            self.transfer_variable.random_padding_cipher_trans_var
        )
        self.aggregate_every_n_epoch = aggregate_every_n_epoch
        self._params: list = []

        self._should_stop = False
        self.metrics_summary = []

    # Todo: 服务器和客户端之间建立连接的部分，可以不用考虑
    def init(self):
        self.random_padding_cipher.create_cipher()

    def encrypt(self, tensor: torch.Tensor, weight):
        return self.random_padding_cipher.encrypt(
            torch.clone(tensor).detach().mul_(weight)
        ).numpy()

    # Todo: 发送模型
    #  tensors是模型参数，weight是模型聚合权重
    def send_model(self, tensors, bn_data, weight, scene_info):
        tensor_arrs = []
        for tensor in tensors:
            tensor_arr = tensor.data.cpu().numpy()
            tensor_arrs.append(tensor_arr)
        bn_arrs = []
        for bn_item in bn_data:
            bn_arr = bn_item.data.cpu().numpy()
            bn_arrs.append(bn_arr)
        self.aggregator.send_model(
            (tensor_arrs, bn_arrs, weight, scene_info), suffix=self._suffix()
        )

    # 接收模型
    def recv_model(self):
        return self.aggregator.get_aggregated_model(suffix=self._suffix())

    # Todo: 向服务器发送相关的度量指标
    def send_metrics(self, metrics, weight):
        self.aggregator.send_model((metrics, weight), suffix=self._suffix(group="metrics"))

    def recv_convergence(self):
        return self.aggregator.get_aggregated_model(
            suffix=self._suffix(group="convergence")
        )

    # 发送、接收全局模型并更新本地模型
    def do_aggregation(self, bn_data, weight, scene_info, device):
        # 发送全局模型
        self.send_model(self._params, bn_data, weight, scene_info)
        LOGGER.warn(f"{self.aggregation_iteration}个模型发送完毕")

        recv_elements: typing.List = self.recv_model()
        LOGGER.warn("模型接收完毕")
        global_model, bn_data = recv_elements
        # 使用接收的全局模型更新本地模型
        agg_tensors = []
        for arr in global_model:
            agg_tensors.append(torch.from_numpy(arr).to(device))
        for param, agg_tensor in zip(self._params, agg_tensors):
            # Todo: param.grad处理的是哪种情况
            if param.grad is None:
                continue
            param.data.copy_(agg_tensor)

        bn_tensors = []
        for arr in bn_data:
            bn_tensors.append(torch.from_numpy(arr).to(device))
        return bn_tensors

    # 关于度量的向量
    def do_convergence_check(self, weight, metrics):
        self.send_metrics(metrics, weight)
        # 接收收敛情况
        # return self.recv_convergence()
        return False

    # 配置聚合参数，将优化器中的参数提取出来
    def configure_aggregation_params(self, optimizer):
        if optimizer is not None:
            self._params = [
                param
                # 不是完全倒序，对于嵌套for循环，先声明的在前面
                for param_group in optimizer.param_groups[0:7]  # 这里不考虑scene_linear的参数，因为关于场景的类别是不一致的
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
    # Todo: 这里的name关系到FATE架构的通信，不能随便更改
    def __init__(self, max_num_aggregation, eps=0.0, name="default"):
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

    def recv_metrics(self):
        return self.aggregator.get_models(suffix=self._suffix(group="metrics"))

    def do_convergence_check(self):
        loss_metrics_pairs = self.recv_metrics()
        total_metrics = None
        total_weight = 0.0

        for metrics, weight in loss_metrics_pairs:
            cur_metrics = [metric * weight for metric in metrics]
            if total_metrics is None:
                total_metrics = cur_metrics
            else:
                total_metrics = [x + y for x, y in zip(total_metrics, cur_metrics)]  # Todo: 这样是附加，而不是对应位置相加
            total_weight += weight

        # 这里的除也要改
        mean_metrics = [metric / total_weight for metric in total_metrics]

        avgloss_writer.writerow([self.aggregation_iteration] + mean_metrics)

        mean_loss = mean_metrics[-1]

        is_converged = abs(mean_loss - self._loss) < self._eps

        self._loss = mean_metrics[-1]
        LOGGER.info(f"convergence check: loss={mean_loss}, is_converged={is_converged}")
        return is_converged, mean_loss


def build_aggregator(param: GCNParam, init_iteration=0):
    context = FedServerContext(
        max_num_aggregation=param.max_iter,
        eps=param.early_stop_eps
    )
    context.init(init_aggregation_iteration=init_iteration)
    fed_aggregator = GCNFedAggregator(context)
    return fed_aggregator


def build_fitter(param: GCNParam, train_data, valid_data):
    category_dir = '/data/projects/fate/my_practice/dataset/coco/'

    # Todo: [WARN]
    # param.batch_size = 4
    # param.max_iter = 1000
    # param.num_labels = 80
    # param.device = 'cuda:0'
    # param.lr = 0.0001
    # category_dir = '/home/klaus125/research/fate/my_practice/dataset/coco'

    epochs = param.aggregate_every_n_epoch * param.max_iter
    context = FedClientContext(
        max_num_aggregation=param.max_iter,
        aggregate_every_n_epoch=param.aggregate_every_n_epoch
    )
    # 与服务器进行握手
    context.init()
    inp_name = 'coco_glove_word2vec.pkl'
    # 构建数据集

    batch_size = param.batch_size
    dataset_loader = DatasetLoader(category_dir, train_data.path, valid_data.path, inp_name=inp_name)

    # Todo: 图像规模减小
    train_loader, valid_loader = dataset_loader.get_loaders(batch_size)

    fitter = GCNFitter(param, epochs, context=context)
    return fitter, train_loader, valid_loader


class GCNFedAggregator(object):
    def __init__(self, context: FedServerContext):
        self.context = context
        self.model = None
        self.bn_data = None

    def fit(self, loss_callback):
        while not self.context.finished():
            recv_elements: typing.List[typing.Tuple] = self.context.recv_model()
            cur_iteration = self.context.aggregation_iteration
            LOGGER.warn(f'收到{len(recv_elements)}个客户端发送过来的模型')
            tensors = [party_tuple[0] for party_tuple in recv_elements]
            bn_tensors = [party_tuple[1] for party_tuple in recv_elements]

            degrees = [party_tuple[2] for party_tuple in recv_elements]
            self.bn_data = aggregate_bn_data(bn_tensors, degrees)

            self.model = aggregate_whole_model(tensors, degrees)
            LOGGER.warn(f'当前聚合轮次为:{cur_iteration}，聚合完成，准备向客户端分发模型')

            self.context.send_model((self.model, self.bn_data))
            LOGGER.warn(f'当前聚合轮次为:{cur_iteration}，模型参数分发成功！')

            self.context.do_convergence_check()

            self.context.increase_aggregation_iteration()

        if self.context.finished():
            print(os.getcwd())
            np.save('global_model', self.model)
            np.save('bn_data', self.bn_data)

    def export_model(self, param):
        pass

    @classmethod
    def load_model(cls, model_obj, meta_obj, param):
        param.restore_from_pb(meta_obj.params)

    pass

    @classmethod
    def load_model(cls, model_obj, meta_obj, param):
        pass

    @staticmethod
    def dataset_align():
        LOGGER.info("start label alignment")
        label_mapping = HomoLabelEncoderArbiter().label_alignment()
        LOGGER.info(f"label aligned, mapping: {label_mapping}")


# Todo: 对gcn fitter的改写
class GCNFitter(object):
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
        self.context = context
        self.label_mapping = label_mapping

        # Todo: 现有的gcn分类器
        self.model, self.scheduler, self.optimizer, self.gcn_optimizer = _init_gcn_learner(self.param,
                                                                                           self.param.device)

        # 使用非对称损失
        self.criterion = AsymmetricLossOptimized().to(self.param.device)

        # self.criterion = torch.nn.MultiLabelSoftMarginLoss().to(self.param.device)

        self.start_epoch, self.end_epoch = 0, epochs

        # 聚合策略的相关参数
        # 1. 按照训练使用的总样本数进行聚合
        self._num_data_consumed = 0
        # Todo: 以下两种参数需要知道确切的标签信息，因此，在训练的批次迭代中进行更新
        # 2. 按照训练使用的标签数进行聚合，对应FLAG论文
        self._num_label_consumed = 0
        # 3. 按照每个标签所包含的样本数进行聚合，维护一个list，对应Partial Supervised论文
        self._num_per_labels = [0] * self.param.num_labels

        # Todo: 初始化平均精度度量器
        self.ap_meter = AveragePrecisionMeter(difficult_examples=False)

        self.lr_scheduler = None
        self.gcn_lr_scheduler = None
        self.epoch_scene_cnts = None

    def get_label_mapping(self):
        return self.label_mapping

    # 执行拟合操作
    def fit(self, train_loader, valid_loader):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.on_fit_epoch_start(epoch, len(train_loader.sampler))
            valid_metrics = self.train_validate(epoch, train_loader, valid_loader, self.scheduler)
            self.on_fit_epoch_end(epoch, valid_loader, valid_metrics)
            if self.context.should_stop():
                break

    def on_fit_epoch_start(self, epoch, num_samples):
        if self._all_consumed_data_aggregated:
            self._num_data_consumed = num_samples
            self._all_consumed_data_aggregated = False
        else:
            self._num_data_consumed += num_samples
        # 初始化epoch_scene_cnts
        self.epoch_scene_cnts = None

    def on_fit_epoch_end(self, epoch, valid_loader, valid_metrics):
        metrics = valid_metrics
        if self.context.should_aggregate_on_epoch(epoch):
            self.aggregate_model(epoch)
            status = self.context.do_convergence_check(
                len(valid_loader.sampler), metrics
            )
            if status:
                self.context.set_converged()
            self._all_consumed_data_aggregated = True

            self._num_data_consumed = 0
            self._num_label_consumed = 0
            self._num_per_labels = [0] * self.param.num_labels

            self.context.increase_aggregation_iteration()

    # 执行拟合逻辑的编写
    def train_one_epoch(self, epoch, train_loader, scheduler):
        # 度量重置
        self.ap_meter.reset()
        # Todo: 调整学习率的部分放到scheduler中执行
        metrics = self.train(train_loader, self.model, self.criterion, self.optimizer, epoch, self.param.device,
                             scheduler)
        return metrics

    def validate_one_epoch(self, epoch, valid_loader, scheduler):
        self.ap_meter.reset()
        metrics = self.validate(valid_loader, self.model, self.criterion, epoch, self.param.device, scheduler)
        return metrics

    def aggregate_model(self, epoch):
        # 配置参数，将优化器optimizer中的参数写入到list中
        self.context.configure_aggregation_params(self.optimizer)
        # 调用上下文执行聚合
        # 发送模型并接收聚合后的模型

        # bn_data添加
        bn_data = []
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                bn_data.append(layer.running_mean)
                bn_data.append(layer.running_var)

        # Partial Supervised聚合策略
        weight_list = list(self._num_per_labels)
        weight_list.append(self._num_data_consumed)

        # 包装一个scene_info，包括场景分类器和每个场景下的邻接矩阵
        scene_info = ()

        # FedAvg聚合策略
        agg_bn_data = self.context.do_aggregation(weight=weight_list, bn_data=bn_data,
                                                  scene_info=scene_info,
                                                  device=self.param.device)
        idx = 0
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.running_mean.data.copy_(agg_bn_data[idx])
                idx += 1
                layer.running_var.data.copy_(agg_bn_data[idx])
                idx += 1

    def train_validate(self, epoch, train_loader, valid_loader, scheduler):
        self.train_one_epoch(epoch, train_loader, scheduler)
        valid_metrics = None
        if valid_loader:
            valid_metrics = self.validate_one_epoch(epoch, valid_loader, scheduler)
        if self.scheduler:
            self.scheduler.on_epoch_end(epoch, self.optimizer)
        return valid_metrics

    def train(self, train_loader, model, criterion, optimizer, epoch, device, scheduler):
        total_samples = len(train_loader.sampler)
        batch_size = 1 if total_samples < train_loader.batch_size else train_loader.batch_size
        steps_per_epoch = math.ceil(total_samples / batch_size)

        model.train()
        # Todo: 记录损失的相关信息
        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        ENTROPY_LOSS_KEY = 'Entropy Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter()),
                              (ENTROPY_LOSS_KEY, tnt.AverageValueMeter())])

        sigmoid_func = torch.nn.Sigmoid()  # 非对称损失中需要传入sigmoid之后的值
        for train_step, ((features, inp), target) in enumerate(train_loader):
            # features是图像特征，inp是输入的标签相关性矩阵
            features = features.to(device)
            target = target.to(device)
            inp = inp.to(device)

            self._num_per_labels += target.t().sum(dim=1).cpu().numpy()

            # 也可在聚合时候统计，这里为明了起见，直接统计
            self._num_label_consumed += target.sum().item()

            # 计算模型输出
            # Todo: 这里还要传入target以计算熵函数
            output = model(features, inp, y=target)
            predicts = output['output']
            entropy_loss = output['entropy_loss']
            batch_scene_cnts = output['scene_cnts']
            # 将scene_cnts加到该epoch的scene_cnts中
            num_scenes = len(batch_scene_cnts)
            if self.epoch_scene_cnts is None:
                self.epoch_scene_cnts = batch_scene_cnts
            else:
                for i in range(num_scenes):
                    self.epoch_scene_cnts[i] += batch_scene_cnts[i]

            # Todo: 将计算结果添加到ap_meter中
            self.ap_meter.add(predicts.data, target)

            lambda_entropy = 1
            objective_loss = criterion(sigmoid_func(predicts), target)

            overall_loss = objective_loss + lambda_entropy * entropy_loss

            losses[OVERALL_LOSS_KEY].add(overall_loss.item())
            losses[OBJECTIVE_LOSS_KEY].add(objective_loss.item())
            losses[ENTROPY_LOSS_KEY].add(entropy_loss.item())
            optimizer.zero_grad()

            overall_loss.backward()

            optimizer.step()

            # predicts_norm = torch.mean(predicts).item()

            # LOGGER.warn(
            #     f"[train] epoch={epoch}, step={train_step} / {steps_per_epoch},lr={optimizer.param_groups[1]['lr']},"
            #     f"mAP={100 * self.ap_meter.value()[0].item()},loss={overall_loss.item()},predicts_norm={predicts_norm}")

        # Todo: 这里对学习率进行调整
        if (epoch + 1) % 4 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        mAP, _ = self.ap_meter.value()
        mAP *= 100
        loss = losses[OBJECTIVE_LOSS_KEY].mean
        # 这里还统计其他数据

        OP, OR, OF1, CP, CR, CF1 = self.ap_meter.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.ap_meter.overall_topk(3)
        metrics = [OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, mAP.item(), loss]
        train_writer.writerow([epoch] + metrics)

        train_loss_writer.writerow(
            [epoch, losses[OBJECTIVE_LOSS_KEY].mean, losses[ENTROPY_LOSS_KEY].mean, losses[OVERALL_LOSS_KEY].mean])
        return metrics

    def validate(self, valid_loader, model, criterion, epoch, device, scheduler):
        total_samples = len(valid_loader.sampler)
        batch_size = valid_loader.batch_size
        steps = math.ceil(total_samples / batch_size)

        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])
        sigmoid_func = torch.nn.Sigmoid()
        model.eval()
        self.ap_meter.reset()

        with torch.no_grad():
            for validate_step, ((features, inp), target) in enumerate(valid_loader):
                features = features.to(device)
                inp = inp.to(device)
                target = target.to(device)

                output = model(features, inp, y=target)
                predicts = output['output']

                # Todo: 将计算结果添加到ap_meter中
                self.ap_meter.add(predicts.data, target)

                objective_loss = criterion(sigmoid_func(predicts), target)

                losses[OBJECTIVE_LOSS_KEY].add(objective_loss.item())
                # Todo: 这里需要对target进行detach操作吗？
                self.ap_meter.add(predicts.data, target)

        mAP, _ = self.ap_meter.value()
        mAP *= 100
        loss = losses[OBJECTIVE_LOSS_KEY].mean
        OP, OR, OF1, CP, CR, CF1 = self.ap_meter.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.ap_meter.overall_topk(3)
        metrics = [OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, mAP.item(), loss]
        valid_writer.writerow([epoch] + metrics)
        return metrics


def _init_gcn_learner(param, device='cpu'):
    # 使用SALGL模型
    # 每个客户端捕捉到的是不同的场景，因此，用不到adjList了
    # Todo: adjList在多场景条件下的适配
    model = full_salgl(param.pretrained, device)
    gcn_optimizer = None

    lr, lrp = param.lr, 1
    optimizer = torch.optim.SGD(model.get_config_optim(lr=lr, lrp=lrp),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=1e-4)

    scheduler = None
    return model, scheduler, optimizer, gcn_optimizer
