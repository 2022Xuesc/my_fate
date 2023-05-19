# 服务器与客户端的通用逻辑
import copy
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
from federatedml.nn.backend.gcn.models import gcn_resnet101
from federatedml.nn.backend.pytorch.data import COCO
from federatedml.nn.homo_nn import _consts
from federatedml.param.gcn_param import GCNParam
from federatedml.util import LOGGER
from federatedml.framework.homo.blocks import aggregator, random_padding_cipher
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.util.homo_label_encoder import HomoLabelEncoderArbiter

from federatedml.nn.backend.gcn.utils import MultiScaleCrop, Warp

# 统计数据的存储路径
stats_dir = os.path.join(os.getcwd(), 'stats')

if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)

buf_size = 1
# 定义和实验数据记录相关的对象

# Todo: 客户端的相关记录
train_file = open(os.path.join(stats_dir, 'train.csv'), 'w', buffering=buf_size)
train_writer = csv.writer(train_file)
train_writer.writerow(
    ['epoch', 'OP', 'OR', 'OF1', 'CP', 'CR', 'CF1', 'OP_3', 'OR_3', 'OF1_3', 'CP_3', 'CR_3', 'CF1_3', 'map', 'loss'])

loss_file = open(os.path.join(stats_dir, 'loss.csv'), 'w', buffering=buf_size)
loss_writer = csv.writer(loss_file)
loss_writer.writerow(['epoch', 'obj_loss', 'reg_loss', 'overall_loss'])

valid_file = open(os.path.join(stats_dir, 'valid.csv'), 'w', buffering=buf_size)
valid_writer = csv.writer(valid_file)

train_writer.writerow(
    ['epoch', 'OP', 'OR', 'OF1', 'CP', 'CR', 'CF1', 'OP_3', 'OR_3', 'OF1_3', 'CP_3', 'CR_3', 'CF1_3', 'map', 'loss'])

# Todo: 服务器端的相关记录
avgloss_file = open(os.path.join(stats_dir, 'avgloss.csv'), 'w', buffering=buf_size)
avgloss_writer = csv.writer(avgloss_file)
avgloss_writer.writerow(
    ['agg_iter', 'OP', 'OR', 'OF1', 'CP', 'CR', 'CF1', 'OP_3', 'OR_3', 'OF1_3', 'CP_3', 'CR_3', 'CF1_3', 'map', 'loss'])


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
    def send_model(self, tensors, weight):
        tensor_arrs = []
        for tensor in tensors:
            tensor_arr = tensor.data.cpu().numpy()
            tensor_arrs.append(tensor_arr)
        self.aggregator.send_model(
            (tensor_arrs, weight), suffix=self._suffix()
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
    def do_aggregation(self, weight, device):
        # 发送全局模型
        self.send_model(self._params, weight)
        LOGGER.warn(f"{self.aggregation_iteration}个模型发送完毕")

        recv_elements: typing.List = self.recv_model()
        LOGGER.warn("模型接收完毕")

        # 使用接收的全局模型更新本地模型
        agg_tensors = []
        for arr in recv_elements:
            agg_tensors.append(torch.from_numpy(arr).to(device))
        for param, agg_tensor in zip(self._params, agg_tensors):
            # Todo: param.grad处理的是哪种情况
            if param.grad is None:
                continue
            param.data.copy_(agg_tensor)

    # 关于度量的向量
    def do_convergence_check(self, weight, metrics):
        self.metrics_summary.append(metrics)

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
        # send_bytes = 0
        # for aggregated_arr in aggregated_arrs:
        #     send_bytes += aggregated_arr.nbytes
        self.aggregator.send_aggregated_model(aggregated_arrs, suffix=self._suffix())

    # 接收客户端模型
    def recv_model(self):
        return self.aggregator.get_models(suffix=self._suffix())

    # 发送收敛状态
    def send_convergence_status(self, status):
        self.aggregator.send_aggregated_model(
            (status), suffix=self._suffix(group="convergence")
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

        # self.send_convergence_status(is_converged)

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
    epochs = param.aggregate_every_n_epoch * param.max_iter
    context = FedClientContext(
        max_num_aggregation=param.max_iter,
        aggregate_every_n_epoch=param.aggregate_every_n_epoch
    )
    # 与服务器进行握手
    context.init()
    category_dir = '/data/projects/fate/my_practice/dataset/coco/'
    # category_dir = '/home/klaus125/research/fate/my_practice/dataset/coco'

    inp_name = 'coco_glove_word2vec.pkl'

    # 构建数据集
    train_dataset = make_dataset(
        data=train_data,
        transforms=train_transforms(),
        category_dir=category_dir,
        inp_name=inp_name
    )
    valid_dataset = make_dataset(
        data=valid_data,
        transforms=valid_transforms(),
        category_dir=category_dir,
        inp_name=inp_name
    )
    batch_size = param.batch_size
    if batch_size < 0 or len(train_dataset) < batch_size:
        batch_size = len(train_dataset)
    shuffle = False

    drop_last = False
    num_workers = 32

    # Todo: 定义collate_fn
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
        drop_last=drop_last, shuffle=shuffle
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size, num_workers=num_workers,
        drop_last=drop_last, shuffle=shuffle
    )
    fitter = GCNFitter(param, epochs, context=context)
    return fitter, train_loader, valid_loader


def make_dataset(data, category_dir, transforms, inp_name):
    return COCO(data.path, config_dir=category_dir, transforms=transforms, inp_name=inp_name)


class GCNFedAggregator(object):
    def __init__(self, context: FedServerContext):
        self.context = context

    def fit(self, loss_callback):
        while not self.context.finished():
            recv_elements: typing.List[typing.Tuple] = self.context.recv_model()
            cur_iteration = self.context.aggregation_iteration
            LOGGER.warn(f'收到{len(recv_elements)}客户端发送过来的模型')
            tensors = [party_tuple[0] for party_tuple in recv_elements]

            # Todo: 这里还需要分析权重
            degrees = [party_tuple[1] for party_tuple in recv_elements]

            # 如果聚合权重是一个list，则分别聚合
            if isinstance(degrees[0], list):
                self.aggregate_by_labels(tensors, degrees)
            else:
                total_degree = sum(degrees)
                for i in range(len(tensors)):
                    for j, tensor in enumerate(tensors[i]):
                        tensor *= degrees[i]
                        tensor /= total_degree
                        if i != 0:
                            tensors[0][j] += tensor
            LOGGER.warn(f'当前聚合轮次为:{cur_iteration}，聚合完成，准备向客户端分发模型')

            self.context.send_model(tensors[0])
            LOGGER.warn(f'当前聚合轮次为:{cur_iteration}，模型参数分发成功！')
            is_converged, loss = self.context.do_convergence_check()
            loss_callback(self.context.aggregation_iteration, float(loss))
            self.context.increase_aggregation_iteration()
            if is_converged:
                break

    # 分标签聚合
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
        self.model, self.scheduler, self.optimizer = _init_gcn_learner(self.param, self.param.device)

        self.criterion = torch.nn.MultiLabelSoftMarginLoss()

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
        self.ap_meter = AveragePrecisionMeter(difficult_examples=True)

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

        # FedAvg聚合策略
        self.context.do_aggregation(weight=self._num_data_consumed, device=self.param.device)

        # Flag聚合策略
        # self.context.do_aggregation(weight=self._num_label_consumed, device=self.param.device)

        # Partial Supervised聚合策略
        # weight_list = list(self._num_per_labels)
        # weight_list.append(self._num_data_consumed)
        # self.context.do_aggregation(weight=weight_list, device=self.param.device)

    def train_validate(self, epoch, train_loader, valid_loader, scheduler):
        self.train_one_epoch(epoch, train_loader, scheduler)
        valid_metrics = None
        if valid_loader:
            valid_metrics = self.validate_one_epoch(epoch, valid_loader, scheduler)
        if self.scheduler:
            self.scheduler.on_epoch_end(epoch, self.optimizer)
        return valid_metrics

    def validate(self, valid_loader, model, criterion, epoch, device, scheduler):
        total_samples = len(valid_loader.sampler)
        # batch_size = 1用于本地测试
        batch_size = 1 if total_samples < valid_loader.batch_size else valid_loader.batch_size
        steps = math.ceil(total_samples / batch_size)

        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])
        model.eval()
        with torch.no_grad():
            for validate_step, ((features, inp), target) in enumerate(valid_loader):
                features = features.to(device)
                inp = inp.to(device)
                target = target.to(device)

                output = model(features, inp)
                loss = criterion(output, target)
                # Todo: 这里需要对target进行detach操作吗？
                self.ap_meter.add(output.data, target)

                if scheduler:
                    agg_loss = scheduler.before_backward_pass(epoch, loss, return_loss_components=True)
                    loss = agg_loss.overall_loss
                    losses[OVERALL_LOSS_KEY].add(loss.item())
                    for lc in agg_loss.loss_components:
                        if lc.name not in losses:
                            losses[lc.name] = tnt.AverageValueMeter()
                        losses[lc.name].add(lc.value.item())
                else:
                    losses[OVERALL_LOSS_KEY].add(loss.item())
                LOGGER.warn(
                    f'[valid] epoch = {epoch}：{validate_step} / {steps},loss={loss.item()}')
        map = 100 * self.ap_meter.value().mean()
        loss = losses[OVERALL_LOSS_KEY].mean
        OP, OR, OF1, CP, CR, CF1 = self.ap_meter.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.ap_meter.overall_topk(3)
        metrics = [OP, OR, OF1, CP, CR, CF1, OP_k, OF1_k, CP_k, CR_k, CF1_k, map.item(), loss]
        # 将这些指标写入文件中
        # 'OP', 'OR', 'OF1', 'CP', 'CR', 'CF1', 'OP_3', 'OR_3', 'OF1_3', 'CP_3', 'CR_3', 'CF1_3'
        valid_writer.writerow([epoch] + metrics)
        return metrics

    def train(self, train_loader, model, criterion, optimizer, epoch, device, scheduler):
        total_samples = len(train_loader.sampler)
        batch_size = 1 if total_samples < train_loader.batch_size else train_loader.batch_size
        steps_per_epoch = math.ceil(total_samples / batch_size)

        model.train()
        # Todo: 记录损失的相关信息
        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

        for train_step, ((features, inp), target) in enumerate(train_loader):
            # features是图像特征，inp是输入的标签相关性矩阵
            features = features.to(device)
            inp = inp.to(device)
            target = target.to(device)

            # 计算模型输出
            output = model(features, inp)
            loss = criterion(output, target)

            losses[OBJECTIVE_LOSS_KEY].add(loss.item())

            # Todo: 将计算结果添加到ap_meter中
            self.ap_meter.add(output.data, target)

            if scheduler:
                agg_loss = scheduler.before_backward_pass(epoch, loss, return_loss_components=True)
                loss = agg_loss.overall_loss
                losses[OVERALL_LOSS_KEY].add(loss.item())
                for lc in agg_loss.loss_components:
                    if lc.name not in losses:
                        losses[lc.name] = tnt.AverageValueMeter()
                    losses[lc.name].add(lc.value.item())
            else:
                losses[OVERALL_LOSS_KEY].add(loss.item())

            REG_LOSS_KEY = 'L2Regularizer_loss'
            # 依然使用l2正则化器
            loss_writer.writerow(
                [epoch, losses['Objective Loss'].mean, losses[REG_LOSS_KEY].mean if REG_LOSS_KEY in losses else -1,
                 losses[OVERALL_LOSS_KEY].mean])

            # 打印进度，打印进度中只关注损失
            LOGGER.warn(
                f'[train] epoch={epoch}, step={train_step} / {steps_per_epoch},loss={loss.item()}')

            optimizer.zero_grad()
            loss.backward()

            # 移除掉较大的梯度
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)

            optimizer.step()
        # epoch结束后，处理相关的指标
        map = 100 * self.ap_meter.value().mean()
        loss = losses[OVERALL_LOSS_KEY].mean
        OP, OR, OF1, CP, CR, CF1 = self.ap_meter.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.ap_meter.overall_topk(3)
        metrics = [OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, map, loss]
        # 将这些指标写入文件中
        # 'OP', 'OR', 'OF1', 'CP', 'CR', 'CF1', 'OP_3', 'OR_3', 'OF1_3', 'CP_3', 'CR_3', 'CF1_3', 'map', 'loss'
        train_writer.writerow([epoch] + metrics)
        return metrics


def _init_gcn_learner(param, device='cpu'):
    # Todo: 这里将in_channel暂设置为300，之后再写入到param中
    in_channel = 300
    model = gcn_resnet101(param.pretrained, param.dataset, t=param.t, adj_file=param.adj_file,
                          device=param.device, num_classes=param.num_labels, in_channel=in_channel)
    # learning rate和learning rate for pretrained_layers
    # Todo: 这里lrp是预训练层的学习率比例值
    lr, lrp = param.lr, 0.1
    optimizer = torch.optim.SGD(model.get_config_optim(lr, lrp),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    scheduler = None
    if param.sched_dict:
        scheduler = config_scheduler(model, optimizer, param.sched_dict, scheduler)
    return model, scheduler, optimizer


# 这里图像大小采用原论文中的设定
image_size = 448


def train_transforms():
    return transforms.Compose([
        MultiScaleCrop(image_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
        # 将图像进行水平翻转
        transforms.RandomHorizontalFlip(),
        # 转换为张量
        transforms.ToTensor(),
        # 对图像进行归一化，以下两个list分别是RGB通道的均值和标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def valid_transforms():
    return transforms.Compose([
        Warp(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
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

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        if pos_count != 0:
            precision_at_i /= pos_count
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
