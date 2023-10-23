# 服务器与客户端的通用逻辑
import math
import numpy as np
import torch
import torch.nn
import torchnet.meter as tnt
import torchvision.transforms as transforms
from torch.nn.utils.rnn import *

import copy

import copy
import csv
import os
import typing
from collections import OrderedDict
from federatedml.framework.homo.blocks import aggregator, random_padding_cipher
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.nn.backend.multi_label.config import config_scheduler
from federatedml.nn.backend.multi_label.models import *

from federatedml.nn.backend.pytorch.data import COCO
from federatedml.param.multi_label_param import MultiLabelParam
from federatedml.util import LOGGER
from federatedml.util.homo_label_encoder import HomoLabelEncoderArbiter

from federatedml.nn.backend.multi_label.losses.AsymmetricLoss import *
# 导入依赖图相关的包
from federatedml.nn.backend.multi_label.prunners.depgraph.dependency import *
from federatedml.nn.backend.multi_label.prunners.depgraph.function import *
from federatedml.nn.backend.multi_label.prunners.depgraph.importance import *
from federatedml.nn.backend.multi_label.prunners.depgraph.meta_pruner import *
from federatedml.nn.backend.multi_label.prunners.depgraph.specific_channel_pruner import *

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
        self._masks: list = []

        self._should_stop = False
        self.loss_summary = []

        # Todo: 选择传输的参数设置
        # 深层的传输间隔
        self.deep_theta = 0
        # 上一次深层传输的回合数量
        self.last_transmission_iter = 0
        # 选择比例随训练进度衰减的权重，该值越大，衰减得越厉害
        self.lambda_k = 1
        # 保存接收到的全局模型，用于验证保留较小权重的效果
        self.last_global_model = None

    def init(self):
        self.random_padding_cipher.create_cipher()

    def encrypt(self, tensor: torch.Tensor, weight):
        return self.random_padding_cipher.encrypt(
            torch.clone(tensor).detach().mul_(weight)
        ).numpy()

    # 发送模型
    # 这里tensors是模型参数，weight是模型聚合权重
    def send_model(self, tensors, masks, weight):
        tensor_arrs = []
        for tensor in tensors:
            # 说明是替换后的层
            if len(tensor) == 0:
                tensor_arrs.append([])
                continue
            tensor_arr = tensor.data.cpu().numpy()
            tensor_arrs.append(tensor_arr)

        mask_arrs = []
        for mask in masks:
            if len(mask) == 0:
                mask_arrs.append([])
                continue
            mask_arr = mask.data.cpu().numpy()
            mask_arrs.append(mask_arr)
        self.aggregator.send_model(
            (tensor_arrs, mask_arrs, weight), suffix=self._suffix()
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
    def do_aggregation(self, weight, device):
        # 发送全局模型
        self.send_model(self._params, self._masks, weight)
        LOGGER.warn("模型发送完毕")

        recv_elements: typing.List = self.recv_model()
        # Todo: 将全局模型和本地的训练进度合并
        #  关注self._masks中0元素所在的位置
        # 直接覆盖
        LOGGER.warn("模型接收完毕")
        agg_tensors = []
        for arr in recv_elements:
            agg_tensors.append(torch.from_numpy(arr).to(device))
        # Todo: 记录接收到的全局模型，便于比较
        self.last_global_model = agg_tensors
        for param, agg_tensor in zip(self._params, agg_tensors):
            if param.grad is None:
                continue
            param.data.copy_(agg_tensor)

    def do_convergence_check(self, weight, mAP, loss_value):
        self.loss_summary.append(loss_value)
        self.send_loss(mAP, loss_value, weight)
        return self.recv_loss()

    # 计算层传输率，这对于每一层来说是相等的
    def calculate_layer_ratio(self):
        # Todo: 对lambda_k的选取进行修改
        # return 1 / (self.lambda_k * self.aggregation_iteration + 1)
        sparsities = [0, 0.07, 0.14, 0.2, 0.26, 0.33, 0.39, 0.46, 0.55, 0.66]
        if self.aggregation_iteration < len(sparsities):
            return sparsities[self.aggregation_iteration]
        else:
            return 0

    # 配置聚合参数，将优化器中的参数提取出来
    # Todo: 在这里应用选择传输的算法
    def configure_aggregation_params(self, dep_model, optimizer):
        layer_ratio = self.calculate_layer_ratio()
        # 获取优化器中的参数列表
        self._params = [
            param
            # 不是完全倒序，对于嵌套for循环，先声明的在前面
            for param_group in optimizer.param_groups
            for param in param_group["params"]
        ]

        # Todo: 使用模型计算masks，因此，无需拷贝
        # self._params2server = copy.deepcopy(self._params)
        # 先对层进行筛选
        layers_num = len(self._params)

        # Todo: 传输所有层
        LOGGER.warn(f"回合 {self.aggregation_iteration}时，总的参数传输率为{layer_ratio}")

        # select_list = [True for i in range(layers_num)]
        # select_layers(self._params2server, select_list=select_list)

        # 返回值是每一层的布尔矩阵
        # 已经对self._params进行了修改，保留变化最大的前p部分参数，将其余参数置为0
        self._masks, layer_ratios,total_ratio = drop_channels_from_person(dep_model)
        # 至此，self._params已经配置完成，将其和self._selected_list一起发送给服务器端
        LOGGER.warn(f"回合 {self.aggregation_iteration}时，总传输比例为{total_ratio},每层的参数传输率为{layer_ratios}")
        # Todo: 计算一下总的传输比例

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


class SyncAggregator(object):
    def __init__(self, context: FedServerContext):
        self.context = context
        self.model = None

    def fit(self, loss_callback):
        while not self.context.finished():
            # Todo: 这里应该是同步接收的方式
            recv_elements: typing.List[typing.Tuple] = self.context.recv_model()

            cur_iteration = self.context.aggregation_iteration
            LOGGER.warn(f'收到{len(recv_elements)}客户端发送过来的模型')

            tensors = [party_tuple[0] for party_tuple in recv_elements]
            masks = [party_tuple[1] for party_tuple in recv_elements]
            degrees = [party_tuple[2] for party_tuple in recv_elements]
            # 对tensors进行重新组合
            self.replace_tensors(tensors, masks)
            # 分标签进行聚合
            self.aggregate_by_labels(tensors, degrees)

            self.model = tensors[0]
            LOGGER.warn(f'当前聚合轮次为:{cur_iteration}，聚合完成，准备向客户端分发模型')
            # Todo: 这里不仅要发送全局模型，还要发送聚合的总权重以及最后一层中各个类的总权重

            self.context.send_model(tensors[0])

            LOGGER.warn(f'当前聚合轮次为:{cur_iteration}，模型参数分发成功！')
            # 还需要进行收敛验证，目的是统计平均结果
            self.context.do_convergence_check()
            # 同步方式下，服务器端也需要记录聚合轮次
            self.context.increase_aggregation_iteration()
        if self.context.finished():
            np.save('global_model', self.model)

    def replace_tensors(self, tensors, masks):
        if self.model is None:
            return
        # 遍历每个客户端
        client_nums = len(tensors)
        for i in range(client_nums):
            layer_nums = len(tensors[i])
            # 遍历每一层参数
            for j in range(layer_nums):
                tensor = tensors[i][j]
                mask = masks[i][j]
                # Todo: 注意处理tensor和mask为空的情况
                if tensor == []:
                    # 使用self.model[j]替代
                    tensors[i][j] = self.model[j]
                else:
                    tensor[np.logical_not(mask)] = self.model[j][np.logical_not(mask)]

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
    param.max_iter = 100
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
    param.batch_size = 1
    param.max_iter = 100
    param.device = 'cuda:0'

    epochs = param.aggregate_every_n_epoch * param.max_iter
    context = FedClientContext(
        max_num_aggregation=param.max_iter,
        aggregate_every_n_epoch=param.aggregate_every_n_epoch
    )
    # 与服务器进行握手
    context.init()

    # 对数据集构建代码的修改

    # 使用绝对路径
    # category_dir = '/data/projects/dataset'
    category_dir = '/home/klaus125/research/fate/my_practice/dataset/coco'

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
        # 使用对称损失
        self.criterion = AsymmetricLossOptimized().to(self.param.device)
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

        # Todo: OneCycleLR调度器的配置
        self.lr_scheduler = None

    def get_label_mapping(self):
        return self.label_mapping

    # 执行拟合操作
    def fit(self, train_loader, valid_loader):

        # 初始化OneCycleLR学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                                max_lr=self.param.lr,
                                                                epochs=self.end_epoch,
                                                                steps_per_epoch=len(train_loader))

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
        # accuracy, loss = self.train_rnn_cnn(train_loader, self.model, self.criterion, self.optimizer,
        #                                     epoch, self.param.device, scheduler)

        train_writer.writerow([epoch, mAP, loss])
        return mAP, loss

    def validate_one_epoch(self, epoch, valid_loader, scheduler):
        mAP, loss = self.validate(valid_loader, self.model, self.criterion, epoch, self.param.device,
                                  scheduler)
        valid_writer.writerow([epoch, mAP, loss])
        return mAP, loss

    def aggregate_model(self, epoch, weight):
        # 配置参数，将优化器optimizer中的参数写入到list中
        # dep_model = self.model.clone()
        dep_model = copy.deepcopy(self.model)
        self.context.configure_aggregation_params(dep_model, self.optimizer)

        # Partial Supervised聚合策略
        weight_list = list(self._num_per_labels)
        weight_list.append(self._num_data_consumed)
        self.context.do_aggregation(weight=weight_list, device=self.param.device)

    def train_validate(self, epoch, train_loader, valid_loader, scheduler):
        mAP, loss = self.train_one_epoch(epoch, train_loader, scheduler)
        if valid_loader:
            mAP, loss = self.validate_one_epoch(epoch, valid_loader, scheduler)
        if self.scheduler:
            self.scheduler.on_epoch_end(epoch, self.optimizer)
        return mAP, loss

    def train(self, train_loader, model, criterion, optimizer, epoch, device, scheduler):
        total_samples = len(train_loader.sampler)
        batch_size = 1 if total_samples < train_loader.batch_size else train_loader.batch_size
        steps_per_epoch = math.ceil(total_samples / batch_size)

        self.ap_meter.reset()
        model.train()

        # 对Loss进行更新
        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])
        # 这里不使用inp
        for train_step, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(device)
            target = target.to(device)

            # Debug看这里的统计是否准确
            self._num_per_labels += target.t().sum(dim=1).cpu().numpy()

            # 也可在聚合时候统计，这里为明了起见，直接统计
            self._num_label_consumed += target.sum().item()

            output = model(inputs)
            self.ap_meter.add(output.data, target.data)

            loss = criterion(output, target)
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


def drop_channels_from_person(dep_model):
    masks = []
    layer_ratios = []
    # dep_model就是需要移除的模型，其参数就是需要移除的参数
    example_inputs = torch.randn(1,3,224,224).to(dep_model.conv1.weight.device)

    layers_to_prune = None
    # json_path = '/home/klaus125/research/fate/fate/python/federatedml/nn/multi_label/data.json'
    json_path = '/data/projects/fate/fate/python/federatedml/nn/multi_label/data.json'

    with open(json_path, 'r') as json_file:
        layers_to_prune = json.load(json_file)
    pruner = SpecificChannelPruner(
        dep_model,
        example_inputs,
        layer_prune_idxs=layers_to_prune
    )
    # name2masks = pruner.step()
    masks = list(pruner.step().values())
    ones = 0
    total_nums = 0
    for j in range(len(masks)):
        layer_mask = masks[j]
        layer_ratios.append((layer_mask.sum() * 1.0 / layer_mask.numel()).item())
        ones += layer_mask.sum()
        total_nums += layer_mask.numel()
    return masks, layer_ratios,ones * 1.0 / total_nums
# 选择传输部分的代码
# 客户端权重、最近全局模型的权重、选取的层比例
# client_weights和客户端的优化器绑定
def save_largest_part_of_weights(dep_model, client_weights, global_weights, layer_ratio):
    # 每一层选择的位置的布尔矩阵
    masks = []
    layer_ratios = []
    if global_weights is None or layer_ratio == 1:
        # 无需mask，直接返回即可
        return masks,layer_ratios
    # 获取依赖模型的参数列表，方便对其进行赋值
    parameters = list(dep_model.parameters())
    # 依次遍历每一层
    for i in range(len(client_weights)):
        # 跳过删除后的层
        if len(client_weights[i]) == 0:
            # 添加占位符
            masks.append(torch.Tensor())
            continue

        # 对client_weights进行原地修改，如果不传输，则将其设定为最近全局模型 -->
        # Todo: 直接设置为0，因为会接收模型，进行聚合。也就是说，不保留较小的训练进度
        with torch.no_grad():
            parameters[i].data.copy_(client_weights[i] - global_weights[i])

    # 拷贝完成后，使用dep graph
    imp = MagnitudeImportance(p=1)
    example_inputs = torch.randn(1, 3, 224, 224).to(dep_model.conv1.weight.device)
    pruner = MetaPruner(
        dep_model,
        example_inputs,
        global_pruning=True,
        importance=imp,
        ch_sparsity=layer_ratio
    )
    masks = list(pruner.step().values())
    for j in range(len(masks)):
        layer_mask = masks[j]
        layer_ratios.append((layer_mask.sum() * 1.0 / layer_mask.numel()).item())

    return masks, layer_ratios


# 确保输入的client_weights和mask都是一维向量
def get_mask(client_weights, percentage):
    mask = torch.zeros(len(client_weights)).to(client_weights.device)
    _, topk_indices = torch.topk(client_weights, k=int(len(client_weights) * percentage))
    # 将掩码矩阵对应的位置设置为1
    mask.scatter_(0, topk_indices, 1)
    return mask


# 选择传输的函数，过滤层
# select_list是传输每个层的布尔向量
def select_layers(client_weights, select_list):
    # 对client_weights进行原地修改
    for i in range(len(client_weights)):
        if select_list[i] is False:
            # 如果不保留第i层，直接清空
            client_weights[i] = torch.Tensor()
