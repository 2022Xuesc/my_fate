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
from federatedml.nn.backend.multi_label.losses.AsymmetricLoss import *
from federatedml.nn.backend.multi_label.models import *
from federatedml.nn.backend.utils.VOC_APMeter import AveragePrecisionMeter
from federatedml.nn.backend.utils.aggregators.aggregator import *
from federatedml.nn.backend.utils.loader.dataset_loader import DatasetLoader
from federatedml.nn.backend.utils.mylogger.mywriter import MyWriter
from federatedml.param.multi_label_param import MultiLabelParam
from federatedml.util import LOGGER
from federatedml.util.homo_label_encoder import HomoLabelEncoderArbiter

cur_dir_name = os.getcwd()
my_writer = MyWriter(dir_name=os.getcwd())
train_header = ['epoch', 'mAP', 'train_loss']
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
        self._params2server: list = []
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
    def send_model(self, tensors, bn_data, masks, weight):
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

        bn_arrs = []
        for bn_item in bn_data:
            bn_arr = bn_item.data.cpu().numpy()
            bn_arrs.append(bn_arr)

        self.aggregator.send_model(
            (tensor_arrs, bn_arrs, mask_arrs, weight), suffix=self._suffix()
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
    # Todo: bn_data照样需要传输，选择传输只对模型参数进行
    def do_aggregation(self, bn_data, weight, device):
        # 发送全局模型
        self.send_model(self._params2server, bn_data, self._masks, weight)
        LOGGER.warn("模型发送完毕")

        recv_elements: typing.List = self.recv_model()
        global_model, bn_data, total_weight_list = recv_elements
        total_weight_list = np.where(total_weight_list == 0, 1, total_weight_list)
        agg_ratio = weight / total_weight_list
        # 直接覆盖
        LOGGER.warn("模型接收完毕")
        agg_tensors = []
        for arr in global_model:
            agg_tensors.append(torch.from_numpy(arr).to(device))
        # Todo: 记录接收到的全局模型，便于比较
        self.last_global_model = agg_tensors
        # Todo: 并不是直接复制
        #  如果layer_ratio == 1，则正常进行聚合
        #  否则layer_ratio < 1，说明有需要保留的
        if len(self._masks) == 0:
            for param, agg_tensor in zip(self._params, agg_tensors):
                if param.grad is None:
                    continue
                param.data.copy_(agg_tensor)
        else:
            self.agg_local_model(agg_tensors, agg_ratio)

        bn_tensors = []
        for arr in bn_data:
            bn_tensors.append(torch.from_numpy(arr).to(device))
        return bn_tensors

    def agg_local_model(self, agg_tensors, agg_ratio):
        # 遍历每个层
        layer_nums = len(self._params)
        # 计算聚合权重

        for i, layer_tensor in enumerate(self._params):
            # 深层当时没有传输，则不必聚合
            # Todo: 深层没有传输，也需要聚合吧？
            #  是应该聚合的，保留深层的本地训练进度
            mask = self._masks[i]
            if len(mask) == 0:  # 整个深层都没有传输，整层都进行平均
                # 如果是最后两层，则分标签聚合
                if i == layer_nums - 2 or i == layer_nums - 1:
                    for j in range(len(layer_tensor)):  # 遍历每一个标签
                        # 第j个标签（向量）
                        layer_tensor[j].data.copy_(
                            agg_ratio[j] * layer_tensor[j].data + (1 - agg_ratio[j]) * agg_tensors[i][j]
                        )
                else:
                    layer_tensor.data.copy_(agg_ratio[-1] * layer_tensor.data +
                                            (1 - agg_ratio[-1]) * agg_tensors[i])
            else:  # mask不为空，说明传输了，则根据mask的内容进行
                # 如果是最后两层
                if i == layer_nums - 2 or i == layer_nums - 1:
                    # 遍历每一个分类器分量
                    for j in range(len(layer_tensor)):
                        # layer_tensor[j]是实数还是一维向量
                        if len(layer_tensor[j].shape) == 0:
                            layer_tensor[j].data.copy_(self.get_updated_val(layer_tensor[j],
                                                                            agg_tensors[i][j],
                                                                            agg_ratio[j],
                                                                            mask[j]))
                        else:  # 两维的
                            # Todo:也可以直接用torch.where实现
                            for k in range(len(layer_tensor[j])):
                                layer_tensor[j][k].data.copy_(self.get_updated_val(layer_tensor[j][k],
                                                                                   agg_tensors[i][j][k],
                                                                                   agg_ratio[j],
                                                                                   mask[j][k]))
                # 如果是之前的特征层
                else:
                    # layer_tensor[0][0][0][1] * 0.5 + agg_tensors[0][0][0][1] * 0.5
                    # 每层向量是layer_tensor
                    # mask为0的位置设置为聚合值，mask为1的位置设置为全局值
                    layer_tensor.data.copy_(
                        torch.where(mask == 0, agg_ratio[-1] * layer_tensor.data + (1 - agg_ratio[-1]) * agg_tensors[i],
                                    agg_tensors[i]))

    def get_updated_val(self, local_val, global_val, ratio, mask):
        if mask == 1:
            return global_val  # 传输并聚合过了，直接复制
        else:
            return ratio * local_val + (1 - ratio) * global_val  # 否则

    def do_convergence_check(self, weight, mAP, loss_value):
        self.loss_summary.append(loss_value)
        self.send_loss(mAP, loss_value, weight)
        return self.recv_loss()

    # 计算层传输率，这对于每一层来说是相等的
    def calculate_layer_ratio(self):
        # Todo: 对lambda_k的选取进行修改
        return 1 / (self.lambda_k * self.aggregation_iteration + 1)

    # 配置聚合参数，将优化器中的参数提取出来
    # Todo: 在这里应用选择传输的算法
    def configure_aggregation_params(self, optimizer):
        layer_ratio = self.calculate_layer_ratio()
        # 获取优化器中的参数列表
        self._params = [
            param
            # 不是完全倒序，对于嵌套for循环，先声明的在前面
            for param_group in optimizer.param_groups
            for param in param_group["params"]
        ]
        # Todo: 对self._params进行拷贝
        self._params2server = copy.deepcopy(self._params)
        # 先对层进行筛选
        layers_num = len(self._params2server)
        # 根据训练进度判断是否传深层
        if self.last_transmission_iter + self.deep_theta == self.aggregation_iteration:
            select_list = [True for i in range(layers_num)]
            self.deep_theta += 1
            self.last_transmission_iter = self.aggregation_iteration
            LOGGER.warn(f"回合 {self.aggregation_iteration} 传输所有层")
        else:
            # select_list = [i + 1 <= layers_num / 2 for i in range(layers_num)]
            # Todo: 注意这里对于深浅层的划分，check一下71是否合适？
            select_list = [i <= 71 for i in range(layers_num)]
            LOGGER.warn(f"回合 {self.aggregation_iteration} 只传输浅层")
        LOGGER.warn(f"每层的参数传输率为{layer_ratio}")
        select_layers(self._params2server, select_list=select_list)
        # 返回值是每一层的布尔矩阵
        # 已经对self._params进行了修改，保留变化最大的前p部分参数，将其余参数置为0
        self._masks = save_largest_part_of_weights(self._params2server, self.last_global_model, layer_ratio)
        # 至此，self._params已经配置完成，将其和self._selected_list一起发送给服务器端

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


def build_aggregator(param: MultiLabelParam, init_iteration=0):
    context = FedServerContext(
        max_num_aggregation=param.max_iter,
        eps=param.early_stop_eps
    )
    context.init(init_aggregation_iteration=init_iteration)
    # Todo: 这里设置同步或异步的聚合方式
    fed_aggregator = SyncAggregator(context)
    return fed_aggregator


def build_fitter(param: MultiLabelParam, train_data, valid_data):
    dataset = 'voc_expanded'

    # Todo: [WARN]
    # param.batch_size = 2
    # param.max_iter = 1000
    # param.num_labels = 20
    # param.device = 'cuda:0'
    # param.lr = 0.0001
    # param.aggregate_every_n_epoch = 1

    # category_dir = f'/home/klaus125/research/fate/my_practice/dataset/{dataset}'
    category_dir = f'/data/projects/fate/my_practice/dataset/{dataset}'

    epochs = param.aggregate_every_n_epoch * param.max_iter
    context = FedClientContext(
        max_num_aggregation=param.max_iter,
        aggregate_every_n_epoch=param.aggregate_every_n_epoch
    )
    # 与服务器进行握手
    context.init()

    batch_size = param.batch_size
    dataset_loader = DatasetLoader(category_dir, train_data.path, valid_data.path)
    train_loader, valid_loader = dataset_loader.get_loaders(batch_size, dataset="VOC", drop_last=False)

    fitter = MultiLabelFitter(param, epochs, context=context)
    return fitter, train_loader, valid_loader, 'normal'


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
            bn_tensors = [party_tuple[1] for party_tuple in recv_elements]
            masks = [party_tuple[2] for party_tuple in recv_elements]
            degrees = [party_tuple[3] for party_tuple in recv_elements]

            self.bn_data = aggregate_bn_data(bn_tensors, degrees)

            # 对tensors进行重新组合
            self.replace_tensors(tensors, masks)
            # 分标签进行聚合
            # Todo: 先用FedAvg
            self.model = aggregate_whole_model(tensors, degrees)
            # self.model = aggregate_by_labels(tensors, degrees)

            LOGGER.warn(f'当前聚合轮次为:{cur_iteration}，聚合完成，准备向客户端分发模型')
            self.context.send_model((self.model, self.bn_data, np.array(degrees).sum(axis=0)))

            LOGGER.warn(f'当前聚合轮次为:{cur_iteration}，模型参数分发成功！')
            # 还需要进行收敛验证，目的是统计平均结果
            # self.context.do_convergence_check()
            # 同步方式下，服务器端也需要记录聚合轮次
            np.save(f'{cur_dir_name}/global_model_{self.context.aggregation_iteration}', self.model)
            np.save(f'{cur_dir_name}/bn_data_{self.context.aggregation_iteration}', self.bn_data)
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
                    tensors[i][j] = np.copy(self.model[j])
                else:
                    tensor[np.logical_not(mask)] = self.model[j][np.logical_not(mask)]

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
        #  对于VOC来说
        self.ap_meter = AveragePrecisionMeter(difficult_examples=True)

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

    # Todo: 聚合依赖数据的更新
    def on_fit_epoch_start(self, epoch, num_samples):
        if self._all_consumed_data_aggregated:
            self._num_data_consumed = num_samples
            self._all_consumed_data_aggregated = False
        else:
            self._num_data_consumed += num_samples

    def on_fit_epoch_end(self, epoch, valid_loader, valid_metrics):
        if self.context.should_aggregate_on_epoch(epoch):
            self.aggregate_model(epoch)
            # 同步模式下，需要发送loss和mAP
            # mean_mAP, status = self.context.do_convergence_check(
            #     self._num_data_consumed, mAP, loss
            # )
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
        self.ap_meter.reset()
        mAP, ap, loss = self.train(train_loader, self.model, self.criterion, self.optimizer,
                                   epoch, self.param.device, scheduler)
        train_writer.writerow([epoch, mAP, loss])
        train_aps_writer.writerow(ap)
        return loss

    def validate_one_epoch(self, epoch, valid_loader, scheduler):
        mAP, ap, loss = self.validate(valid_loader, self.model, self.criterion, epoch, self.param.device,
                                      scheduler)
        valid_writer.writerow([epoch, mAP, loss])
        valid_aps_writer.writerow(ap)
        return mAP, loss

    def aggregate_model(self, epoch):
        # 配置参数，将优化器optimizer中的参数写入到list中
        self.context.configure_aggregation_params(self.optimizer)

        bn_data = []
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                bn_data.append(layer.running_mean)
                bn_data.append(layer.running_var)

        # Partial Supervised聚合策略
        weight_list = list(self._num_per_labels)
        weight_list.append(self._num_data_consumed)
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

        # 对Loss进行更新
        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])
        # 这里不使用inp
        sigmoid_func = torch.nn.Sigmoid()
        for train_step, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(device)
            prev_target = target.clone()

            target[target == 0] = 1
            target[target == -1] = 0
            target = target.to(device)

            # Debug看这里的统计是否准确
            self._num_per_labels += target.t().sum(dim=1).cpu().numpy()

            # 也可在聚合时候统计，这里为明了起见，直接统计
            self._num_label_consumed += target.sum().item()

            output = model(inputs)
            self.ap_meter.add(output.data, prev_target)

            loss = criterion(sigmoid_func(output), target)
            losses[OBJECTIVE_LOSS_KEY].add(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 4 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

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
        sigmoid_func = torch.nn.Sigmoid()

        model.eval()
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


def _init_learner(param, device='cpu'):
    # Todo: 将通用部分提取出来
    model = create_resnet101_model(param.pretrained, device=device, num_classes=param.num_labels)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=1e-4)
    scheduler = None
    return model, scheduler, optimizer


# 选择传输部分的代码
# 客户端权重、最近全局模型的权重、选取的层比例
# client_weights和客户端的优化器绑定
def save_largest_part_of_weights(client_weights, global_weights, layer_ratio):
    # 每一层选择的位置的布尔矩阵
    masks = []
    if global_weights is None or layer_ratio == 1:
        # 无需mask，直接返回即可
        return masks
    # 依次遍历每一层
    for i in range(len(client_weights)):
        # 跳过删除后的层
        if len(client_weights[i]) == 0:
            # 添加占位符
            masks.append(torch.Tensor())
            continue
        # 需要对参数的形状进行判定
        layer_shape = client_weights[i].shape
        # 将其展平成一维向量
        client_weights[i] = client_weights[i].flatten()
        # 对global_weights也进行reshape操作
        global_weights[i] = global_weights[i].flatten()
        weights_diff = torch.abs(client_weights[i] - global_weights[i])
        # 获取最大的p部分的布尔矩阵
        mask = get_mask(weights_diff, layer_ratio)
        # 对client_weights进行原地修改，如果不传输，则将其设定为最近全局模型 -->
        # Todo: 直接设置为0，因为会接收模型，进行聚合。也就是说，不保留较小的训练进度
        with torch.no_grad():
            client_weights[i].mul_(mask)
        # 对client_weights和mask进行reshape
        client_weights[i] = client_weights[i].reshape(layer_shape)
        # Todo: 这一步可能不太需要，先保留
        global_weights[i] = global_weights[i].reshape(layer_shape)
        mask = mask.reshape(layer_shape)

        masks.append(mask)
    return masks


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
