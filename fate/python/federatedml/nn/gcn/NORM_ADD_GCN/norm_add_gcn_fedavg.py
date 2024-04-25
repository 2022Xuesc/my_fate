import math
import torch
import torch.nn
import torchnet.meter as tnt

import copy
import json
import os
import typing
from collections import OrderedDict
from federatedml.framework.homo.blocks import aggregator, random_padding_cipher
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.nn.backend.gcn.models import *
from federatedml.nn.backend.multi_label.losses.AsymmetricLoss import *
from federatedml.nn.backend.utils.VOC_APMeter import AveragePrecisionMeter
from federatedml.nn.backend.utils.aggregators.aggregator import *
from federatedml.nn.backend.utils.loader.dataset_loader import DatasetLoader
from federatedml.nn.backend.utils.mylogger.mywriter import MyWriter
from federatedml.param.gcn_param import GCNParam
from federatedml.util import LOGGER
from federatedml.util.homo_label_encoder import HomoLabelEncoderArbiter

my_writer = MyWriter(dir_name=os.getcwd())
train_header = ['epoch', 'mAP', 'asym_loss', 'dynamic_adj_loss', 'overall_loss']
valid_header = ['epoch', 'mAP', 'loss']

train_writer = my_writer.get("train.csv", header=train_header)
valid_writer = my_writer.get("valid.csv", header=valid_header)
avgloss_writer = my_writer.get("avgloss.csv", header=['agg_iter', 'mAP', 'avgloss'])

# Todo: 记录每个标签的ap值
train_aps_writer = my_writer.get("train_aps.csv")
valid_aps_writer = my_writer.get("valid_aps.csv")
agg_ap_writer = my_writer.get("agg_ap.csv")


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
    def send_model(self, tensors, bn_data, weight):
        tensor_arrs = []
        for tensor in tensors:
            tensor_arr = tensor.data.cpu().numpy()
            tensor_arrs.append(tensor_arr)
        bn_arrs = []
        for bn_item in bn_data:
            bn_arr = bn_item.data.cpu().numpy()
            bn_arrs.append(bn_arr)
        self.aggregator.send_model(
            (tensor_arrs, bn_arrs, weight), suffix=self._suffix()
        )

    def recv_model(self):
        return self.aggregator.get_aggregated_model(suffix=self._suffix())

    # 接收模型
    def send_metrics(self, ap, mAP, loss, weight):
        self.aggregator.send_model((ap, mAP, loss, weight), suffix=self._suffix(group="metrics"))

    # 发送、接收全局模型并更新本地模型
    def do_aggregation(self, bn_data, weight, device):
        # 发送全局模型
        self.send_model(self._params, bn_data, weight)
        LOGGER.warn("模型发送完毕")

        recv_elements: typing.List = self.recv_model()
        LOGGER.warn("模型接收完毕")
        global_model, bn_data = recv_elements
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
        return bn_tensors

    def do_convergence_check(self, weight, ap, mAP, loss_value):
        self.send_metrics(ap, mAP, loss_value, weight)
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

    def recv_metrics(self):
        return self.aggregator.get_models(suffix=self._suffix(group="metrics"))

    def do_convergence_check(self):
        loss_weight_pairs = self.recv_metrics()
        total_loss = 0.0
        total_weight = 0.0
        total_mAP = 0.

        num_labels = len(loss_weight_pairs[0][0])
        # 对每个标签有贡献的平均精度值
        agg_ap = torch.zeros(num_labels)
        # 对每个标签有贡献的客户端数量
        agg_weight = torch.zeros(num_labels)

        for ap, mAP, loss, weight in loss_weight_pairs:
            for i in range(num_labels):
                if ap[i] != -1:
                    agg_ap[i] += ap[i] * weight
                    agg_weight[i] += weight
            total_loss += loss * weight
            total_mAP += mAP * weight
            total_weight += weight

        for i in range(num_labels):
            # 判断agg_weight是否等于0，如果是，则将其设置成-1，表示没有对应的样本
            if agg_weight[i] == 0:
                agg_ap[i] = -1
            else:
                agg_ap[i] /= agg_weight[i]

        agg_ap_writer.writerow(agg_ap.tolist())
        mean_loss = total_loss / total_weight
        mean_mAP = total_mAP / total_weight

        avgloss_writer.writerow([self.aggregation_iteration, mean_mAP, mean_loss])

        is_converged = abs(mean_loss - self._loss) < self._eps
        self._loss = mean_loss

        # self.send_convergence_status(mean_mAP, is_converged)

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
    # dataset = 'coco'
    # dataset = 'nuswide'
    dataset = 'voc_expanded'

    # category_dir = f'/home/klaus125/research/fate/my_practice/dataset/{dataset}'
    category_dir = f'/data/projects/fate/my_practice/dataset/{dataset}'

    # Todo: [WARN]
    # param.batch_size = 2
    # param.max_iter = 1000
    # param.num_labels = 20
    # param.device = 'cuda:0'
    # param.lr = 0.0001
    # param.aggregate_every_n_epoch = 1

    epochs = param.aggregate_every_n_epoch * param.max_iter
    context = FedClientContext(
        max_num_aggregation=param.max_iter,
        aggregate_every_n_epoch=param.aggregate_every_n_epoch
    )
    # 与服务器进行握手
    context.init()
    inp_name = 'voc_expanded_glove_word2vec.pkl'
    # 构建数据集

    batch_size = param.batch_size
    dataset_loader = DatasetLoader(category_dir, train_data.path, valid_data.path, inp_name=inp_name)

    # Todo: 图像规模减小
    train_loader, valid_loader = dataset_loader.get_loaders(batch_size, dataset="VOC", drop_last=False)

    fitter = GCNFitter(param, epochs, context=context)
    return fitter, train_loader, valid_loader


class GCNFedAggregator(object):
    def __init__(self, context: FedServerContext):
        self.context = context
        self.model = None
        self.bn_data = None
        self.relation_matrix = None

    def fit(self, loss_callback):
        while not self.context.finished():
            recv_elements: typing.List[typing.Tuple] = self.context.recv_model()
            cur_iteration = self.context.aggregation_iteration
            LOGGER.warn(f'收到{len(recv_elements)}个客户端发送过来的模型')
            tensors = [party_tuple[0] for party_tuple in recv_elements]
            bn_tensors = [party_tuple[1] for party_tuple in recv_elements]

            degrees = [party_tuple[2] for party_tuple in recv_elements]
            self.bn_data = aggregate_bn_data(bn_tensors, degrees)
            # Todo: 这里需要再改改
            #  没有分类层了，因此，无法使用FPSL了

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

        image_id2labels = json.load(open(self.param.adj_file, 'r'))
        num_labels = self.param.num_labels
        adjList = np.zeros((num_labels, num_labels))
        nums = np.zeros(num_labels)
        for image_info in image_id2labels:
            labels = []
            for i in range(num_labels):
                if image_info['labels'][i] != -1:
                    labels.append(i)
                    nums[i] += 1
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

        # 使用非对称的
        # 将adjList主对角线上的数字设置为1
        for i in range(num_labels):
            adjList[i][i] = 1

        self.adjList = adjList

        # Todo: 现有的gcn分类器
        self.model, self.scheduler, self.optimizer, self.gcn_optimizer = _init_gcn_learner(self.param,
                                                                                           self.param.device,
                                                                                           self.adjList)

        # 使用非对称损失
        self.criterion = AsymmetricLossOptimized().to(self.param.device)
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

        self.lr_scheduler = None
        self.gcn_lr_scheduler = None

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
        aps, mAP, loss = valid_metrics
        if self.context.should_aggregate_on_epoch(epoch):
            self.aggregate_model(epoch)
            status = self.context.do_convergence_check(
                self._num_data_consumed, aps, mAP, loss
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
        mAP, ap, asym_loss, dynamic_adj_loss, overall_loss = self.train(train_loader, self.model, self.criterion,
                                                                        self.optimizer, epoch, self.param.device,
                                                                        scheduler)
        train_writer.writerow([epoch, mAP, asym_loss, dynamic_adj_loss, overall_loss])
        train_aps_writer.writerow(ap)
        return overall_loss

    def validate_one_epoch(self, epoch, valid_loader, scheduler):
        self.ap_meter.reset()
        mAP, ap, loss = self.validate(valid_loader, self.model, self.criterion, epoch, self.param.device, scheduler)
        valid_writer.writerow([epoch, mAP, loss])
        valid_aps_writer.writerow(ap)
        return ap, mAP, loss

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

        # FedAvg聚合策略
        agg_bn_data = self.context.do_aggregation(weight=weight_list, bn_data=bn_data,
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
        self.ap_meter.reset()
        model.train()
        # Todo: 记录损失的相关信息
        ASYM_LOSS = 'Asym Loss'
        DYNAMIC_ADJ_LOSS = 'Dynamic Adj Loss'
        OVERALL_LOSS_KEY = 'Overall Loss'
        losses = OrderedDict([(ASYM_LOSS, tnt.AverageValueMeter()),
                              (DYNAMIC_ADJ_LOSS, tnt.AverageValueMeter()),
                              (OVERALL_LOSS_KEY, tnt.AverageValueMeter())])

        sigmoid_func = torch.nn.Sigmoid()

        for train_step, ((features, inp), target) in enumerate(train_loader):
            # features是图像特征，inp是输入的标签相关性矩阵
            features = features.to(device)

            prev_target = target.clone()

            target[target == 0] = 1
            target[target == -1] = 0
            target = target.to(device)

            self._num_per_labels += target.t().sum(dim=1).cpu().numpy()

            # 也可在聚合时候统计，这里为明了起见，直接统计
            self._num_label_consumed += target.sum().item()

            # 计算模型输出
            out1, out2, dynamic_adj_loss = model(features)
            # 求平均
            predicts = (out1 + out2) / 2

            # Todo: 将计算结果添加到ap_meter中
            self.ap_meter.add(predicts.data, prev_target)

            # 非对称损失需要经过sigmoid

            lambda_dynamic = 1
            asym_loss = criterion(sigmoid_func(predicts), target)
            overall_loss = asym_loss + \
                           lambda_dynamic * dynamic_adj_loss

            losses[OVERALL_LOSS_KEY].add(overall_loss.item())
            losses[DYNAMIC_ADJ_LOSS].add(dynamic_adj_loss.item())
            losses[ASYM_LOSS].add(asym_loss.item())
            optimizer.zero_grad()

            overall_loss.backward()
            # Todo: 这里需要对模型的参数进行裁剪吗？
            optimizer.step()

        # Todo: 这里对学习率进行调整
        if (epoch + 1) % 4 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        mAP, ap = self.ap_meter.value()
        mAP *= 100
        overall_loss = losses[OVERALL_LOSS_KEY].mean
        dynamic_adj_loss = losses[DYNAMIC_ADJ_LOSS].mean
        asym_loss = losses[ASYM_LOSS].mean
        return mAP.item(), ap, asym_loss, dynamic_adj_loss, overall_loss

    def validate(self, valid_loader, model, criterion, epoch, device, scheduler):
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

                prev_target = target.clone()
                target[target == 0] = 1
                target[target == -1] = 0
                target = target.to(device)

                out1, out2, _ = model(features)
                # Todo: 将计算结果添加到ap_meter中
                predicts = (out1 + out2) / 2
                self.ap_meter.add(predicts.data, prev_target)

                objective_loss = criterion(sigmoid_func(predicts), target)

                losses[OBJECTIVE_LOSS_KEY].add(objective_loss.item())
        mAP, ap = self.ap_meter.value()
        mAP *= 100
        loss = losses[OBJECTIVE_LOSS_KEY].mean
        return mAP.item(), ap, loss


def _init_gcn_learner(param, device='cpu', adjList=None):
    # in_channel是标签嵌入向量的初始（输入）维度
    # Todo: 对于static_graph优化变量形式，输入通道设置为1024
    #  对于初始化的，使用300即可
    in_channel = 1024
    model = norm_add_gcn_resnet101(param.pretrained, adjList,
                                   device=param.device, num_classes=param.num_labels, in_channels=in_channel,
                                   needOptimize=True)
    gcn_optimizer = None

    lr, lrp = param.lr, 0.1
    optimizer = torch.optim.AdamW(model.get_config_optim(lr=lr, lrp=lrp),
                                  lr=lr,
                                  weight_decay=1e-4)

    scheduler = None
    return model, scheduler, optimizer, gcn_optimizer
