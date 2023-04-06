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

from federatedml.nn.backend.multi_label.config import config_scheduler
from federatedml.nn.backend.multi_label.models import create_model
from federatedml.nn.backend.pytorch.data import MultiLabelDataSet
from federatedml.nn.homo_nn import _consts
from federatedml.param.multi_label_param import MultiLabelParam
from federatedml.util import LOGGER
from federatedml.framework.homo.blocks import aggregator, random_padding_cipher
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.util.homo_label_encoder import HomoLabelEncoderArbiter

log_dir = 'logs'
stats_dir = os.path.join(os.getcwd(), 'stats')
LOGGER.warn(stats_dir)
models_dir = 'models'

if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

fate_logger = logging.getLogger('fate_logger')
arbiter_logger = logging.getLogger('arbiter_logger')

buf_size = 1
# 定义和实验数据记录相关的对象

train_file = open(os.path.join(stats_dir, 'train.csv'), 'w', buffering=buf_size)
train_writer = csv.writer(train_file)
train_writer.writerow(['epoch', 'precision', 'recall', 'train_loss'])

loss_file = open(os.path.join(stats_dir, 'loss.csv'), 'w', buffering=buf_size)
loss_writer = csv.writer(loss_file)
loss_writer.writerow(['epoch', 'obj_loss', 'reg_loss', 'overall_loss'])
valid_file = open(os.path.join(stats_dir, 'valid.csv'), 'w', buffering=buf_size)
valid_writer = csv.writer(valid_file)
valid_writer.writerow(['epoch', 'precision', 'recall', 'valid_loss'])

avgloss_file = open(os.path.join(stats_dir, 'avgloss.csv'), 'w', buffering=buf_size)
avgloss_writer = csv.writer(avgloss_file)
avgloss_writer.writerow(['agg_iter', 'precision', 'recall', 'avgloss'])


class _FedBaseContext(object):
    def __init__(self, max_num_aggregation, name):
        self.max_num_aggregation = max_num_aggregation
        # Todo: 关于名称的指定？
        self._name = name
        self._aggregation_iteration = 0

    # Todo: 发送消息时，可以指定当前的迭代信息，考虑在此处进行异步优化
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
        self.loss_summary = []

    def init(self):
        self.random_padding_cipher.create_cipher()

    def encrypt(self, tensor: torch.Tensor, weight):
        return self.random_padding_cipher.encrypt(
            torch.clone(tensor).detach().mul_(weight)
        ).numpy()

    # 发送模型
    # 这里tensors是模型参数，weight是模型聚合权重
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

    def send_loss(self, precision, recall, loss, weight):
        self.aggregator.send_model((precision, recall, loss, weight), suffix=self._suffix(group="loss"))

    def recv_loss(self):
        return self.aggregator.get_aggregated_model(
            suffix=self._suffix(group="convergence")
        )

    # 发送、接收全局模型并更新本地模型
    def do_aggregation(self, weight, device):
        # 发送全局模型
        self.send_model(self._params, weight)
        LOGGER.warn("模型发送完毕")

        recv_elements: typing.List = self.recv_model()
        LOGGER.warn("模型接收完毕")
        agg_tensors = []
        for arr in recv_elements:
            agg_tensors.append(torch.from_numpy(arr).to(device))
        for param, agg_tensor in zip(self._params, agg_tensors):
            # param.grad处理的是哪种情况
            if param.grad is None:
                continue
            param.data.copy_(agg_tensor)

    def do_convergence_check(self, weight, precision, recall, loss_value):
        self.loss_summary.append(loss_value)
        self.send_loss(precision, recall, loss_value, weight)
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
    def send_convergence_status(self, precision, recall, status):
        self.aggregator.send_aggregated_model(
            (precision, recall, status), suffix=self._suffix(group="convergence")
        )

    def recv_losses(self):
        return self.aggregator.get_models(suffix=self._suffix(group="loss"))

    def do_convergence_check(self):
        arbiter_logger.info('正在等待客户端发送loss')
        loss_weight_pairs = self.recv_losses()
        arbiter_logger.info('成功接受客户端发送的loss')
        total_loss = 0.0
        total_weight = 0.0
        total_precision = 0.0
        total_recall = 0.0

        for precision, recall, loss, weight in loss_weight_pairs:
            total_loss += loss * weight
            total_precision += precision * weight
            total_recall += recall * weight
            total_weight += weight
        mean_loss = total_loss / total_weight
        mean_precision = total_precision / total_weight
        mean_recall = total_recall / total_weight

        avgloss_writer.writerow([self.aggregation_iteration, mean_precision, mean_recall, mean_loss])

        is_converged = abs(mean_loss - self._loss) < self._eps

        self.send_convergence_status(mean_precision, mean_recall, is_converged)

        self._loss = mean_loss
        arbiter_logger.info(f'收敛性验证：loss={mean_loss},is_converged={is_converged}')
        LOGGER.info(f"convergence check: loss={mean_loss}, is_converged={is_converged}")
        return is_converged, mean_loss


def build_aggregator(param: MultiLabelParam, init_iteration=0):
    context = FedServerContext(
        max_num_aggregation=param.max_iter,
        eps=param.early_stop_eps
    )
    context.init(init_aggregation_iteration=init_iteration)
    fed_aggregator = MultiLabelFedAggregator(context)
    return fed_aggregator


def build_fitter(param: MultiLabelParam, train_data, valid_data):
    epochs = param.aggregate_every_n_epoch * param.max_iter
    context = FedClientContext(
        max_num_aggregation=param.max_iter,
        aggregate_every_n_epoch=param.aggregate_every_n_epoch
    )
    # 与服务器进行握手
    context.init()
    expected_label_type = np.int64

    # 构建数据集
    train_dataset = make_dataset(
        data=train_data,
        transforms=train_transforms(),
    )
    valid_dataset = make_dataset(
        data=valid_data,
        transforms=valid_transforms(),
    )
    batch_size = param.batch_size
    if batch_size < 0:
        batch_size = len(train_dataset)
    shuffle = False

    drop_last = True
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


def make_dataset(data, transforms):
    return MultiLabelDataSet(data.path, transforms)


class MultiLabelFedAggregator(object):
    def __init__(self, context: FedServerContext):
        self.context = context

    def fit(self, loss_callback):
        while not self.context.finished():
            recv_elements: typing.List[typing.Tuple] = self.context.recv_model()
            cur_iteration = self.context.aggregation_iteration
            arbiter_logger.info(f'当前聚合轮次为{cur_iteration},成功接收到客户端的模型参数!')
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
                        tensor[k] *= degrees[i][k]
                        if degrees_sum[k] == 0:
                            continue
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


class MultiLabelFitter(object):
    def __init__(
            self,
            param,
            epochs,
            label_mapping=None,
            context: FedClientContext = None
    ):
        self.param = copy.deepcopy(param)
        self._all_consumed_data_aggregated = True
        self.best_precision = 0
        self.context = context
        self.label_mapping = label_mapping
        (self.model, self.scheduler, self.optimizer) = _init_learner(self.param, self.param.device)
        self.criterion = torch.nn.BCELoss().to(self.param.device)
        self.start_epoch, self.end_epoch = 0, epochs

        # 聚合策略的相关参数
        # 1. 按照训练使用的总样本数进行聚合
        self._num_data_consumed = 0
        # Todo: 以下两种参数需要知道确切的标签信息，因此，在训练的批次迭代中进行更新
        # 2. 按照训练使用的标签数进行聚合，对应FLAG论文
        self._num_label_consumed = 0
        # 3. 按照每个标签所包含的样本数进行聚合，维护一个list，对应Partial Supervised论文
        self._num_per_labels = [0] * self.param.num_labels

    def get_label_mapping(self):
        return self.label_mapping

    # 执行拟合操作
    def fit(self, train_loader, valid_loader):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.on_fit_epoch_start(epoch, len(train_loader.sampler))
            precision, recall, loss = self.train_validate(epoch, train_loader, valid_loader, self.scheduler)
            fate_logger.info(f'已完成一轮训练+验证')
            LOGGER.warn(f'epoch={epoch}/{self.end_epoch},precision={precision},recall={recall},loss={loss}')
            self.on_fit_epoch_end(epoch, valid_loader, precision, recall, loss)
            if self.context.should_stop():
                break

    def on_fit_epoch_start(self, epoch, num_samples):
        if self._all_consumed_data_aggregated:
            fate_logger.info(f'新一轮聚合开始')
            self._num_data_consumed = num_samples
            self._all_consumed_data_aggregated = False
        else:
            self._num_data_consumed += num_samples
        fate_logger.info(f'当前epoch为{epoch}，已经消耗的样本数量为{self._num_data_consumed}')

    def on_fit_epoch_end(self, epoch, valid_loader, valid_precision, valid_recall, valid_loss):
        precision = valid_precision
        recall = valid_recall
        loss = valid_loss
        if self.context.should_aggregate_on_epoch(epoch):
            self.aggregate_model(epoch)
            mean_precision, mean_recall, status = self.context.do_convergence_check(
                len(valid_loader.sampler), precision, recall, loss
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
        precision, recall, loss = self.train(train_loader, self.model, self.criterion, self.optimizer,
                                             epoch, self.param.device, scheduler)
        fate_logger.info(f'训练阶段结束,epoch = {epoch},平均loss = {loss}')
        train_writer.writerow([epoch, precision, recall, loss])
        return precision, recall, loss

    def validate_one_epoch(self, epoch, valid_loader, scheduler):
        precision, recall, loss = validate(valid_loader, self.model, self.criterion, epoch, self.param.device,
                                           scheduler)
        fate_logger.info(f'验证阶段结束,epoch = {epoch},precision = {precision},recall = {recall},平均loss = {loss}')
        valid_writer.writerow([epoch, precision, recall, loss])
        return precision, recall, loss

    def aggregate_model(self, epoch):
        # 配置参数，将优化器optimizer中的参数写入到list中
        self.context.configure_aggregation_params(self.optimizer)
        # 调用上下文执行聚合
        # 发送模型并接收聚合后的模型

        # FedAvg聚合策略
        # self.context.do_aggregation(weight=self._num_data_consumed, device=self.param.device)
        # Flag聚合策略
        # self.context.do_aggregation(weight=self._num_label_consumed, device=self.param.device)
        # Partial Supervised聚合策略
        weight_list = list(self._num_per_labels)
        weight_list.append(self._num_data_consumed)
        self.context.do_aggregation(weight=weight_list, device=self.param.device)

    def train_validate(self, epoch, train_loader, valid_loader, scheduler):
        precision, recall, loss = self.train_one_epoch(epoch, train_loader, scheduler)
        if valid_loader:
            precision, recall, loss = self.validate_one_epoch(epoch, valid_loader, scheduler)
        if self.scheduler:
            self.scheduler.on_epoch_end(epoch, self.optimizer)
        return precision, recall, loss

    def train(self, train_loader, model, criterion, optimizer, epoch, device, scheduler):
        total_samples = len(train_loader.sampler)
        batch_size = train_loader.batch_size
        steps_per_epoch = math.ceil(total_samples / batch_size)

        fate_logger.info(f'开始一轮训练，epoch为:{epoch}，batch_size为:{batch_size}，每个epoch需要的step为:{steps_per_epoch}')


        sigmoid_func = torch.nn.Sigmoid()

        model.train()
        precisions = tnt.AverageValueMeter()
        recalls = tnt.AverageValueMeter()

        # 对Loss进行更新
        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

        for train_step, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(device)
            target = target.to(device)

            self._num_per_labels += target.t().sum(dim=1).cpu().numpy()

            # 也可在聚合时候统计，这里为明了起见，之间统计
            self._num_label_consumed += target.sum().item()

            output = model(inputs)
            loss = criterion(sigmoid_func(output), target)
            losses[OBJECTIVE_LOSS_KEY].add(loss.item())
            precision, recall = calculate_accuracy_mode1(sigmoid_func(output), target)
            precisions.add(precision)
            recalls.add(recall)

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

            loss_writer.writerow(
                [epoch, losses['Objective Loss'].mean, losses['L2Regularizer_loss'].mean, losses[OVERALL_LOSS_KEY].mean])

            # 打印进度
            LOGGER.warn(
                f'[train] epoch={epoch}, step={train_step} / {steps_per_epoch},precision={precision},recall={recall},loss={loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return precisions.mean, recalls.mean, losses[OVERALL_LOSS_KEY].mean


def validate(valid_loader, model, criterion, epoch, device, scheduler):
    # 这里
    precisions = tnt.AverageValueMeter()
    recalls = tnt.AverageValueMeter()
    # 对Loss进行更新
    OVERALL_LOSS_KEY = 'Overall Loss'
    OBJECTIVE_LOSS_KEY = 'Objective Loss'
    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    total_samples = len(valid_loader.sampler)
    batch_size = valid_loader.batch_size

    total_steps = math.ceil(total_samples / batch_size)
    fate_logger.info(f'开始一轮验证，epoch为:{epoch}，batch_size为:{batch_size}，每个epoch需要的step为:{total_steps}')
    sigmoid_func = torch.nn.Sigmoid()
    model.eval()

    with torch.no_grad():
        for validate_step, (inputs, target) in enumerate(valid_loader):
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(sigmoid_func(output), target)
            precision, recall = calculate_accuracy_mode1(sigmoid_func(output), target)
            precisions.add(precision)
            recalls.add(recall)
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
                f'[valid] epoch = {epoch}：{validate_step} / {total_steps},precision={precision},recall={recall},loss={loss}')
    return precisions.mean, recalls.mean, losses[OVERALL_LOSS_KEY].mean


def _init_learner(param, device='cpu'):
    # Todo: 将通用部分提取出来
    model = create_model(param.pretrained, param.dataset, param.arch, num_classes=param.num_labels, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=param.lr, momentum=0.9)
    scheduler = None
    if param.sched_dict:
        scheduler = config_scheduler(model, optimizer, param.sched_dict, scheduler)
    return model, scheduler, optimizer


def calculate_accuracy_mode1(model_pred, labels):
    # 精度阈值
    accuracy_th = 0.5
    pred_res = model_pred > accuracy_th
    pred_res = pred_res.float()
    pred_sum = torch.sum(pred_res)
    if pred_sum == 0:
        return 0, 0
    target_sum = torch.sum(labels)
    # element-wise multiply
    true_predict_sum = torch.sum(pred_res * labels)
    # 模型预测的结果中有多少个结果正确
    precision = true_predict_sum / pred_sum
    # 模型预测正确的结果中，占样本真实标签的数量
    recall = true_predict_sum / target_sum
    return precision.item(), recall.item()


def train_transforms():
    return transforms.Compose([
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
    ])


def valid_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
