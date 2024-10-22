# 服务器与客户端的通用逻辑
import math
import torch.nn
import torchnet.meter as tnt

import copy
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

cur_dir_name = os.getcwd()
my_writer = MyWriter(dir_name=cur_dir_name)

client_header = ['epoch', 'mAP', 'loss']

train_writer = my_writer.get("train.csv", header=client_header)
valid_writer = my_writer.get("valid.csv", header=client_header)
avgloss_writer = my_writer.get("avgloss.csv", header=['agg_iter', 'mAP', 'avgloss'])

# Todo: 记录每个标签的ap值
train_aps_writer = my_writer.get("train_aps.csv")
valid_aps_writer = my_writer.get("valid_aps.csv")
agg_ap_writer = my_writer.get("agg_ap.csv")

scene_cnts_writer = my_writer.get("total_scene_cnts.csv")


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
        self.name = name
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
        self.name = self.random_padding_cipher.create_cipher()

    def encrypt(self, tensor: torch.Tensor, weight):
        return self.random_padding_cipher.encrypt(
            torch.clone(tensor).detach().mul_(weight)
        ).numpy()

    def tensor2arr(self, tensors):
        tensor_arrs = []
        for tensor in tensors:
            tensor_arr = tensor.data.cpu().numpy()
            tensor_arrs.append(tensor_arr)
        return tensor_arrs

    # Todo: 发送模型
    #  tensors是模型参数，weight是模型聚合权重
    def send_model(self, tensors, bn_data, weight):
        tensor_arrs = self.tensor2arr(tensors)
        bn_arrs = self.tensor2arr(bn_data)
        self.aggregator.send_model(
            (tensor_arrs, bn_arrs, weight), suffix=self._suffix()
        )

    # 接收模型
    def recv_model(self):
        return self.aggregator.get_aggregated_model(suffix=self._suffix())

    # Todo: 向服务器发送相关的度量指标
    def send_metrics(self, ap, mAP, loss, weight):
        self.aggregator.send_model((ap, mAP, loss, weight), suffix=self._suffix(group="metrics"))

    def recv_convergence(self):
        return self.aggregator.get_aggregated_model(
            suffix=self._suffix(group="convergence")
        )

    # 发送、接收全局模型并更新本地模型
    def do_aggregation(self, bn_data, weight, device):
        # 发送全局模型
        self.send_model(self._params, bn_data, weight)

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
    def do_convergence_check(self, weight, ap, mAP, loss_value):
        self.send_metrics(ap, mAP, loss_value, weight)
        # 接收收敛情况
        # return self.recv_convergence()
        return False

    # 配置聚合参数，将优化器中的参数提取出来
    def configure_aggregation_params(self, optimizer):
        if optimizer is not None:
            self._params = [
                param
                # 不是完全倒序，对于嵌套for循环，先声明的在前面
                for param_group in optimizer.param_groups[0:6]  # 这里不考虑scene_linear的参数，因为关于场景的类别是不一致的
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
    # Todo: [WARN]
    param.batch_size = 2
    param.max_iter = 1000
    param.num_labels = 20
    param.device = 'cuda:0'
    param.lr = 0.0001
    param.aggregate_every_n_epoch = 1

    # category_dir = '/data/projects/fate/my_practice/dataset/voc2012/'
    category_dir = '/home/klaus125/research/fate/my_practice/dataset/voc2012'

    epochs = param.aggregate_every_n_epoch * param.max_iter
    context = FedClientContext(
        max_num_aggregation=param.max_iter,
        aggregate_every_n_epoch=param.aggregate_every_n_epoch
    )
    # 与服务器进行握手
    context.init()
    # 构建数据集
    inp_name = 'voc2012_glove_word2vec.pkl'

    batch_size = param.batch_size
    dataset_loader = DatasetLoader(category_dir, train_data.path, valid_data.path, inp_name=inp_name)

    # Todo: 图像规模减小
    train_loader, valid_loader = dataset_loader.get_loaders(batch_size, dataset='VOC')

    fitter = GCNFitter(param, epochs, context=context)
    return fitter, train_loader, valid_loader, 'normal'


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

            np.save(f'{cur_dir_name}/global_model_{self.context.aggregation_iteration}', self.model)
            np.save(f'{cur_dir_name}/bn_data_{self.context.aggregation_iteration}', self.bn_data)

            self.context.increase_aggregation_iteration()

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
        self.epoch_scene_cnts = None

    def get_label_mapping(self):
        return self.label_mapping

    # 执行拟合操作
    def fit(self, train_loader, valid_loader, agg_type):

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
        aps, mAP, loss = valid_metrics
        if self.context.should_aggregate_on_epoch(epoch):
            self.aggregate_model(epoch)

            self._num_data_consumed = 0
            self._num_label_consumed = 0
            self._num_per_labels = [0] * self.param.num_labels

            self.context.increase_aggregation_iteration()

    # 执行拟合逻辑的编写
    def train_one_epoch(self, epoch, train_loader, scheduler):
        # 度量重置
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

        scene_cnts_writer.writerow([epoch] + self.model.total_scene_cnts)

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
            inp = inp.to(device)

            prev_target = target.clone()

            target[target == 0] = 1
            target[target == -1] = 0
            target = target.to(device)

            self._num_per_labels += target.t().sum(dim=1).cpu().numpy()

            # 也可在聚合时候统计，这里为明了起见，直接统计
            self._num_label_consumed += target.sum().item()

            # 计算模型输出
            # Todo: 这里还要传入target以计算熵函数
            output = model(features, inp, y=target)
            predicts = output['output']
            entropy_loss = output['entropy_loss']

            # Todo: 将计算结果添加到ap_meter中
            self.ap_meter.add(predicts.data, prev_target)

            lambda_entropy = 1
            objective_loss = criterion(sigmoid_func(predicts), target)

            overall_loss = objective_loss + lambda_entropy * entropy_loss

            losses[OVERALL_LOSS_KEY].add(overall_loss.item())
            losses[OBJECTIVE_LOSS_KEY].add(objective_loss.item())
            losses[ENTROPY_LOSS_KEY].add(entropy_loss.item())
            optimizer.zero_grad()

            overall_loss.backward()

            optimizer.step()

        # Todo: 这里对学习率进行调整
        if (epoch + 1) % 4 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        mAP, ap = self.ap_meter.value()
        mAP *= 100
        loss = losses[OBJECTIVE_LOSS_KEY].mean
        return mAP.item(), ap, loss

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
                inp = inp.to(device)

                prev_target = target.clone()
                target[target == 0] = 1
                target[target == -1] = 0
                target = target.to(device)

                output = model(features, inp, y=target)
                predicts = output['output']

                # Todo: 将计算结果添加到ap_meter中
                self.ap_meter.add(predicts.data, prev_target)

                objective_loss = criterion(sigmoid_func(predicts), target)

                losses[OBJECTIVE_LOSS_KEY].add(objective_loss.item())

        mAP, ap = self.ap_meter.value()
        mAP *= 100
        loss = losses[OBJECTIVE_LOSS_KEY].mean
        return mAP.item(), ap, loss


def _init_gcn_learner(param, device='cpu'):
    # 使用SALGL模型
    # 每个客户端捕捉到的是不同的场景，因此，用不到adjList了
    # Todo: adjList在多场景条件下的适配
    num_scenes = 4
    # 基础学习率调大一点，lrp调小点
    lr, lrp = param.lr, 0.1

    model = resnet_salgl(param.pretrained, device, num_scenes=num_scenes, num_classes=param.num_labels)
    gcn_optimizer = None

    # 使用AdamW优化器试试
    optimizer = torch.optim.AdamW(model.get_config_optim(lr=lr, lrp=lrp), lr=param.lr, weight_decay=1e-4)

    scheduler = None
    return model, scheduler, optimizer, gcn_optimizer
