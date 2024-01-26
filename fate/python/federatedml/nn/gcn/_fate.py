import math
import torch.nn
import torchnet.meter as tnt

import copy
import os
import typing
from collections import OrderedDict
from federatedml.framework.homo.blocks import aggregator, random_padding_cipher
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.nn.backend.communication.models.resnet101 import resnet101
from federatedml.nn.backend.multi_label.losses.AsymmetricLoss import *
from federatedml.nn.backend.multi_label.models import *
from federatedml.nn.backend.utils.VOC_APMeter import AveragePrecisionMeter
from federatedml.nn.backend.utils.aggregators.aggregator import *
from federatedml.nn.backend.utils.loader.dataset_loader import DatasetLoader
from federatedml.nn.backend.utils.mylogger.mywriter import MyWriter
from federatedml.param.multi_label_param import MultiLabelParam
from federatedml.util import LOGGER
from federatedml.util.homo_label_encoder import HomoLabelEncoderArbiter

my_writer = MyWriter(dir_name=os.getcwd())

client_header = ['epoch', 'mAP', 'loss']

train_writer = my_writer.get("train.csv", header=client_header)
valid_writer = my_writer.get("valid.csv", header=client_header)
avgloss_writer = my_writer.get("avgloss.csv", header=['agg_iter', 'mAP', 'avgloss'])

# Todo: 记录每个标签的ap值
train_aps_writer = my_writer.get("train_aps.csv")
valid_aps_writer = my_writer.get("valid_aps.csv")
agg_ap_writer = my_writer.get("agg_ap.csv")


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


def build_aggregator(param: MultiLabelParam, init_iteration=0):
    context = FedServerContext(
        max_num_aggregation=param.max_iter,
        eps=param.early_stop_eps
    )
    context.init(init_aggregation_iteration=init_iteration)
    # Todo: 这里设置同步的聚合方式
    fed_aggregator = SyncAggregator(context)
    return fed_aggregator


def build_fitter(param: MultiLabelParam, train_data, valid_data):
    # dataset = 'coco'
    # dataset = 'nuswide'
    dataset = 'voc_expanded'

    # category_dir = f'/home/klaus125/research/fate/my_practice/dataset/{dataset}'
    category_dir = f'/data/projects/fate/my_practice/dataset/{dataset}'

    # Todo: [WARN]
    # param.batch_size = 1
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
    # 构建数据集

    batch_size = param.batch_size
    dataset_loader = DatasetLoader(category_dir, train_data.path, valid_data.path)

    # Todo: 图像规模减小
    train_loader, valid_loader = dataset_loader.get_loaders(batch_size, dataset='VOC')

    fitter = MultiLabelFitter(param, epochs, context=context)
    return fitter, train_loader, valid_loader


class SyncAggregator(object):
    def __init__(self, context: FedServerContext):
        self.context = context
        self.model = None
        self.bn_data = None

    def fit(self, loss_callback):
        while not self.context.finished():
            # Todo: 这里应该是同步接收的方式
            recv_elements: typing.List[typing.Tuple] = self.context.recv_model()

            cur_iteration = self.context.aggregation_iteration
            LOGGER.warn(f'收到{len(recv_elements)}客户端发送过来的模型')

            tensors = [party_tuple[0] for party_tuple in recv_elements]
            # 还有bn层的统计数据
            bn_tensors = [party_tuple[1] for party_tuple in recv_elements]
            # Todo: 对BN层的统计数据进行处理
            # 抽取出
            degrees = [party_tuple[2] for party_tuple in recv_elements]
            self.bn_data = aggregate_bn_data(bn_tensors, degrees)
            # 聚合整个模型，flag论文也是这种聚合方式，只是degrees的生成方式变化
            aggregate_whole_model(tensors, degrees)

            LOGGER.warn(f'当前聚合轮次为:{cur_iteration}，聚合完成，准备向客户端分发模型')

            self.model = tensors[0]
            self.context.send_model((self.model, self.bn_data))

            LOGGER.warn(f'当前聚合轮次为:{cur_iteration}，模型参数分发成功！')
            # 还需要进行收敛验证，目的是统计平均结果
            self.context.do_convergence_check()
            # 同步方式下，服务器端也需要记录聚合轮次
            self.context.increase_aggregation_iteration()

        if self.context.finished():
            np.save('global_model', self.model)

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
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.param = copy.deepcopy(param)
        self._all_consumed_data_aggregated = True
        self.best_precision = 0
        self.context = context
        self.label_mapping = label_mapping

        # Todo: progressive learning的相关设置
        self.num_stages = 8
        self.iters_per_stage = param.max_iter // self.num_stages
        # Todo: [WARN]
        # self.iters_per_stage = 1

        # Todo: 原始的ResNet101分类器
        self.whole_model = resnet101(self.num_stages, num_classes=param.num_labels)

        # Todo: 使用非对称损失
        self.criterion = AsymmetricLossOptimized().to(self.param.device)

        self.start_epoch, self.end_epoch = 0, epochs

        # 聚合策略的相关参数
        # 1. 按照训练使用的总样本数进行聚合
        self._num_data_consumed = 0
        # Todo: 以下两种参数需要知道确切的标签信息，因此，在训练的批次迭代中进行更新
        #  FLAG论文客户端在聚合时可根据标签列表信息直接计算聚合权重
        #  而PS论文需要将标签出现向量发送给服务器端实现分标签聚合
        # 2. 按照训练集的标签宽度进行聚合，对应FLAG论文中alpha = 0的特殊设定
        self._num_label_consumed = 0
        # 3. 按照每个标签所包含的样本数进行聚合，维护一个list，对应FLAG论文和Partial Supervised论文
        self._num_per_labels = [0] * self.param.num_labels

        # Todo: 创建ap_meter
        self.ap_meter = AveragePrecisionMeter(difficult_examples=True)

        self.lr_scheduler = None

    def get_label_mapping(self):
        return self.label_mapping

    # 执行拟合操作
    def fit(self, train_loader, valid_loader):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.on_fit_epoch_start(epoch, len(train_loader.sampler), train_loader)
            valid_metrics = self.train_validate(epoch, train_loader, valid_loader, self.scheduler)
            self.on_fit_epoch_end(epoch, valid_loader, valid_metrics)
            if self.context.should_stop():
                break

    # Todo: 聚合依赖数据的更新
    # 使用train_loader来进行预热
    def on_fit_epoch_start(self, epoch, num_samples, train_loader):
        if self._all_consumed_data_aggregated:
            self._num_data_consumed = num_samples
            self._all_consumed_data_aggregated = False

            # Todo: 说明是一次新iteration
            cur_iter = self.context.aggregation_iteration
            if cur_iter % self.iters_per_stage == 0:
                stage_id = cur_iter // self.iters_per_stage
                self.model, self.optimizer = get_new_learner(self.whole_model, stage_id, self.model, self.param.lr,
                                                             self.param.device)
                # 在这里对模型的new layer进行一下预热
                self.warmup(train_loader)
        else:
            self._num_data_consumed += num_samples

    def warmup(self, train_loader):
        # 预热数据集是train_loader
        # 预热模型是self.model
        # 损失函数是self.criterion
        # 自己定义对应的优化器
        # warmup所在的设备device

        stage_id = self.whole_model.ind
        if stage_id == 0:  # 如果是第一个子模型，则没有需要freeze的部分，直接返回
            return

        self.model.train()
        # Todo: freeze的几个步骤
        # 1. 固定住基本网络，将固定部分的requires_grad设置为False，避免没有必要的反向传播 # Todo: 同时验证这里的设定对之后的模型的影响
        freezed_list = self.model.enc[:stage_id]
        freezed_list.requires_grad_(False)
        freezed_list.eval()

        warmup_lr = 0.0001
        warmup_epoch = 5
        freeze_optimizer = torch.optim.SGD(params=self.model.latest_parameters(), lr=warmup_lr, momentum=0.9,
                                           weight_decay=1e-4)

        device = self.param.device

        sigmoid_func = torch.nn.Sigmoid()

        for i in range(warmup_epoch):
            for train_step, (inputs, target) in enumerate(train_loader):
                inputs = inputs.to(device)

                target[target == 0] = 1
                target[target == -1] = 0
                target = target.to(device)

                output = self.model(inputs)

                loss = self.criterion(sigmoid_func(output), target)

                freeze_optimizer.zero_grad()
                loss.backward()
                freeze_optimizer.step()

    def on_fit_epoch_end(self, epoch, valid_loader, valid_metrics):
        aps, mAP, loss = valid_metrics
        if self.context.should_aggregate_on_epoch(epoch):
            self.aggregate_model(epoch)
            LOGGER.warn("模型接收完成，准备发送指标")
            # 同步模式下，需要发送loss和mAP
            status = self.context.do_convergence_check(
                self._num_data_consumed, aps, mAP, loss
            )
            LOGGER.warn("指标发送完成")
            if status:
                self.context.set_converged()
            self._all_consumed_data_aggregated = True

            # 将相关指标重置为0
            self._num_data_consumed = 0
            self._num_label_consumed = 0
            self._num_per_labels = [0] * self.param.num_labels

            self.context.increase_aggregation_iteration()

    # 执行拟合逻辑的编写
    def train_one_epoch(self, epoch, train_loader, scheduler):
        self.ap_meter.reset()
        # Todo: 调整学习率的部分放到scheduler中执行
        mAP, ap, loss = self.train(train_loader, self.model, self.criterion, self.optimizer, epoch, self.param.device,
                                   scheduler)
        train_writer.writerow([epoch, mAP, loss])
        train_aps_writer.writerow(ap)
        return ap, mAP, loss

    def validate_one_epoch(self, epoch, valid_loader, scheduler):
        self.ap_meter.reset()
        mAP, ap, loss = self.validate(valid_loader, self.model, self.criterion, epoch, self.param.device, scheduler)
        valid_writer.writerow([epoch, mAP, loss])
        valid_aps_writer.writerow(ap)
        # 并且返回验证集的ap
        return ap, mAP, loss

    def aggregate_model(self, epoch, weight=None):
        # 配置参数，将优化器optimizer中的参数写入到list中
        self.context.configure_aggregation_params(self.optimizer)

        bn_data = []
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):  # Todo: group norm和batch norm的区别
                bn_data.append(layer.running_mean)
                bn_data.append(layer.running_var)

        # FedAvg聚合策略
        agg_bn_data = self.context.do_aggregation(weight=self._num_data_consumed, bn_data=bn_data,
                                                  device=self.param.device)

        # Flag聚合策略
        # Todo: 添加聚合参数
        # 这里计算weight
        # agg_bn_data = self.context.do_aggregation(weight=weight, bn_data=bn_data, device=self.param.device)
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

        self.ap_meter.reset()
        model.train()
        model.requires_grad_(True)

        # 对Loss进行更新
        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

        sigmoid_func = torch.nn.Sigmoid()
        for train_step, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(device)

            prev_target = target.clone()

            target[target == 0] = 1
            target[target == -1] = 0
            target = target.to(device)

            self._num_per_labels += target.t().sum(dim=1).cpu().numpy()

            self._num_label_consumed += target.sum().item()

            output = model(inputs)
            self.ap_meter.add(output.data, prev_target)

            # 这里criterion自然会进行sigmoid操作
            loss = criterion(sigmoid_func(output), target)
            losses[OBJECTIVE_LOSS_KEY].add(loss.item())

            # 打印进度
            # LOGGER.warn(
            #    f'[train] epoch={epoch}, step={train_step} / {steps_per_epoch},loss={loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Todo: 这里对学习率进行调整
        if (epoch + 1) % 4 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
            LOGGER.warn("cur learning rate is ", optimizer.param_groups[0]['lr'])

        mAP, ap = self.ap_meter.value()
        mAP *= 100
        loss = losses[OBJECTIVE_LOSS_KEY].mean
        return mAP.item(), ap, loss

    def validate(self, valid_loader, model, criterion, epoch, device, scheduler):
        # 对Loss进行更新
        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

        total_samples = len(valid_loader.sampler)
        batch_size = valid_loader.batch_size

        total_steps = math.ceil(total_samples / batch_size)

        model.eval()
        sigmoid_func = torch.nn.Sigmoid()
        # Todo: 在开始训练之前，重置ap_meter
        self.ap_meter.reset()
        with torch.no_grad():
            for validate_step, (inputs, target) in enumerate(valid_loader):
                inputs = inputs.to(device)

                prev_target = target.clone()
                target[target == 0] = 1
                target[target == -1] = 0
                target = target.to(device)

                output = model(inputs)
                loss = criterion(sigmoid_func(output), target)
                losses[OBJECTIVE_LOSS_KEY].add(loss.item())

                # 将输出和对应的target加入到ap_meter中
                # Todo: 对相关格式的验证
                self.ap_meter.add(output.data, prev_target)

        mAP, ap = self.ap_meter.value()
        mAP *= 100
        loss = losses[OBJECTIVE_LOSS_KEY].mean
        return mAP.item(), ap, loss


# 传入迭代轮次
# 每个阶段多少个迭代，来确认阶段id，从而确定子模型
# 上一个迭代的模型
# 设备


# Todo: 现在单机环境下测试
def get_new_learner(whole_model, stage_id, last_model, learning_rate=0.1, device='cpu'):
    cur_model = whole_model.get_submodel(stage_id, last_model).to(device)
    # Todo: 学习率重启，重新设置lr为初始lr
    optimizer = torch.optim.Adam(cur_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    return cur_model, optimizer
