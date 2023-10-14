# 服务器与客户端的通用逻辑

import math
import torch.nn
import torchnet.meter as tnt
import torchvision.transforms as transforms

import copy
import json
import os
import typing
from collections import OrderedDict
from federatedml.framework.homo.blocks import aggregator, random_padding_cipher
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.nn.backend.multi_label.losses.AsymmetricLoss import *
from federatedml.nn.backend.multi_label.losses.SmoothLoss import *
from federatedml.nn.backend.multi_label.models import *
from federatedml.nn.backend.pytorch.data import COCO
from federatedml.nn.backend.utils.APMeter import AveragePrecisionMeter
# 导入OMP算法模块
from federatedml.nn.backend.utils.OMP import LabelOMP
from federatedml.nn.backend.utils.aggregators.aggregator import *
from federatedml.nn.backend.utils.loader.dataset_loader import DatasetLoader
from federatedml.nn.backend.utils.mylogger.mywriter import MyWriter
from federatedml.param.multi_label_param import MultiLabelParam
from federatedml.util import LOGGER
from federatedml.util.homo_label_encoder import HomoLabelEncoderArbiter

my_writer = MyWriter(dir_name=os.getcwd())

train_writer = my_writer.get("train.csv", header=['epoch', 'mAP', 'train_loss'])
valid_writer = my_writer.get("valid.csv", header=['epoch', 'mAP', 'valid_loss'])
avgloss_writer = my_writer.get("avgloss.csv", header=['agg_iter', 'mAP', 'avgloss'])

train_aps_writer = my_writer.get("train_aps.csv")
val_aps_writer = my_writer.get("val_aps.csv")
agg_ap_writer = my_writer.get("agg_ap.csv")

# Todo: 只有预测阶段有相关性损失
train_loss_writer = my_writer.get("train_loss.csv", header=['epoch', 'entropy_loss', 'relation_loss', 'overall_loss'])


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

    def send_loss(self, ap, mAP, loss, weight):
        self.aggregator.send_model((ap, mAP, loss, weight), suffix=self._suffix(group="loss"))

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

    def do_convergence_check(self, weight, ap, mAP, loss_value):
        self.loss_summary.append(loss_value)
        self.send_loss(ap, mAP, loss_value, weight)
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

        num_labels = len(loss_weight_pairs[0][0])
        # 对每个标签有贡献的平均精度值
        agg_ap = torch.zeros(num_labels)
        # 对每个标签有贡献的客户端数量
        agg_weight = torch.zeros(num_labels)

        # Todo: 这里还要对每个标签的AP值进行平均
        for ap, mAP, loss, weight in loss_weight_pairs:
            # 遍历每一个标签
            for i in range(num_labels):
                if ap[i] != -1:
                    agg_ap[i] += ap[i] * weight
                    agg_weight[i] += weight
            total_loss += loss * weight
            total_mAP += mAP * weight
            total_weight += weight
        # 对agg_ap进行平均
        for i in range(num_labels):
            # 判断agg_weight是否等于0，如果是，则将其设置成-1，表示没有对应的样本
            if agg_weight[i] == 0:
                agg_ap[i] = -1
            else:
                agg_ap[i] /= agg_weight[i]
        # 将agg_ap写入到文件中
        agg_ap_writer.writerow(agg_ap.tolist())

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
            self.bn_data = aggregate_bn_data(bn_tensors, degrees)
            self.relation_matrix = aggregate_relation_matrix(relation_matrices, degrees)
            # 分标签进行聚合
            aggregate_by_labels(tensors, degrees)

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
            np.save('global_model', self.model)
            np.save('bn_data', self.bn_data)

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
    # param.num_labels = 20

    # 使用绝对路径
    # category_dir = '/data/projects/dataset'
    category_dir = "/data/projects/voc2007"
    # category_dir = '/home/klaus125/research/fate/my_practice/dataset/voc_expanded'

    epochs = param.aggregate_every_n_epoch * param.max_iter
    context = FedClientContext(
        max_num_aggregation=param.max_iter,
        aggregate_every_n_epoch=param.aggregate_every_n_epoch
    )
    # 与服务器进行握手
    context.init()

    # 对数据集构建代码的修改

    batch_size = param.batch_size
    dataset_loader = DatasetLoader(category_dir, train_data.path, valid_data.path)
    train_loader, valid_loader = dataset_loader.get_loaders(batch_size)

    fitter = MultiLabelFitter(param, epochs, context=context)
    return fitter, train_loader, valid_loader


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
        num_labels = self.param.num_labels
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
            ap, mAP, loss = self.train_validate(epoch, train_loader, valid_loader, self.scheduler)
            LOGGER.warn(f'epoch={epoch}/{self.end_epoch},mAP={mAP},loss={loss}')
            self.on_fit_epoch_end(epoch, valid_loader, ap, mAP, loss)
            if self.context.should_stop():
                break

    # Todo: 聚合依赖数据的更新
    def on_fit_epoch_start(self, epoch, num_samples):
        if self._all_consumed_data_aggregated:
            self._num_data_consumed = num_samples
            self._all_consumed_data_aggregated = False
        else:
            self._num_data_consumed += num_samples

    def on_fit_epoch_end(self, epoch, valid_loader, ap, valid_mAP, valid_loss):
        mAP = valid_mAP
        loss = valid_loss
        if self.context.should_aggregate_on_epoch(epoch):
            self.aggregate_model(epoch, self._num_data_consumed)
            # 同步模式下，需要发送loss和mAP
            mean_mAP, status = self.context.do_convergence_check(
                self._num_data_consumed, ap, mAP, loss
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
        mAP, ap, loss = self.train(train_loader, self.model, self.criterion, self.optimizer,
                                   epoch, self.param.device, scheduler)

        train_writer.writerow([epoch, mAP, loss])
        train_aps_writer.writerow(ap)
        # 也记录一下ap值
        return ap, mAP, loss

    def validate_one_epoch(self, epoch, valid_loader, scheduler):
        mAP, ap, loss = self.validate(valid_loader, self.model, self.criterion, epoch, self.param.device,
                                      scheduler)
        valid_writer.writerow([epoch, mAP, loss])
        val_aps_writer.writerow(ap)
        # 并且返回验证集的ap
        return ap, mAP, loss

    def construct_relation_by_matrix(self, matrix):
        num_labels = self.param.num_labels
        self.adjList = [dict() for _ in range(num_labels)]
        self.variables = []
        # 设定阈值th
        th = 0.2
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

        labels_num = self.param.num_labels
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
        ap, mAP, loss = self.train_one_epoch(epoch, train_loader, scheduler)
        if valid_loader:
            ap, mAP, loss = self.validate_one_epoch(epoch, valid_loader, scheduler)
        if self.scheduler:
            self.scheduler.on_epoch_end(epoch, self.optimizer)
        return ap, mAP, loss

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
        ENTROPY_LOSS_KEY = 'Entropy Loss'
        RELATION_LOSS_KEY = 'Relation Loss'
        OVERALL_LOSS_KEY = 'Overall Loss'
        losses = OrderedDict([(ENTROPY_LOSS_KEY, tnt.AverageValueMeter()),
                              (RELATION_LOSS_KEY, tnt.AverageValueMeter()),
                              (OVERALL_LOSS_KEY, tnt.AverageValueMeter())])

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

            # Todo: 是这里的影响吗？
            entropy_loss = criterion(predicts, target)

            # Todo: 还是这里的tensor(0)的影响？
            relation_loss = self.lambda_y * LabelSmoothLoss(relation_need_grad=False)(predicts, predict_similarities,
                                                                                      self.adjList)

            # 初始化标签相关性，先计算标签平滑损失，对相关性进行梯度下降
            loss = entropy_loss + relation_loss

            losses[OVERALL_LOSS_KEY].add(loss.item())
            # losses[ENTROPY_LOSS_KEY].add(entropy_loss.item())
            # losses[RELATION_LOSS_KEY].add(relation_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.lr_scheduler.step()

        # train_loss_writer.writerow(
        #     [epoch, losses[ENTROPY_LOSS_KEY].mean, losses[RELATION_LOSS_KEY].mean, losses[OVERALL_LOSS_KEY].mean])
        # 记录ap数组
        mAP, ap = self.ap_meter.value()
        mAP *= 100
        return mAP.item(), ap, losses[OVERALL_LOSS_KEY].mean

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

        mAP, ap = self.ap_meter.value()
        mAP *= 100
        return mAP.item(), ap, losses[OBJECTIVE_LOSS_KEY].mean


def _init_learner(param, device='cpu'):
    # Todo: 将通用部分提取出来
    model = create_resnet101_model(param.pretrained, device=device, num_classes=param.num_labels)
    # 使用Adam优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=param.lr, weight_decay=1e-4)
    scheduler = None
    return model, scheduler, optimizer
