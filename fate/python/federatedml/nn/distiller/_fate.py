import copy
import logging
import logging.config
import os.path
from collections import OrderedDict

import math
import numpy
import numpy as np
import torch
import torch.optim
import torch.utils
import torch.nn as nn
import torchnet.meter as tnt
import typing
from scipy import sparse
from numpy import random

from federatedml.nn.backend.distiller import PruningPolicy
from federatedml.nn.backend.distiller.apputils import tensor_compressor
from federatedml.nn.backend.distiller.apputils.tensor_compressor import CompressedTensor
from federatedml.nn.backend.distiller.models import create_model
from federatedml.framework.homo.blocks import aggregator, random_padding_cipher
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.nn.backend.pytorch.data import VisionDataSet
from federatedml.nn.homo_nn import _consts
from federatedml.param.distiller_param import DistillerParam
from federatedml.protobuf.generated import nn_model_meta_pb2, nn_model_param_pb2
from federatedml.util import LOGGER
from federatedml.util.homo_label_encoder import HomoLabelEncoderArbiter
import torchvision.transforms as transforms
from federatedml.nn.backend.distiller.config import config_scheduler
import federatedml.nn.backend.distiller as distiller
from federatedml.nn.backend.distiller.data_loggers import *
import federatedml.nn.backend.distiller.utils as utils
import federatedml.nn.backend.distiller.quantization as quantization
import federatedml.nn.backend.distiller.quantization.q_utils as q_utils
import federatedml.nn.backend.distiller.quantization.range_linear as range_linear

import csv

log_dir = 'logs'
stats_dir = os.path.join(os.getcwd(), 'stats')
models_dir = 'models'

# 如果以下目录不存在，则进行创建目录不存在，则进行创建
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
# def init_logger():
# logging_conf = '/home/klaus125/research/Distillered-FATE/my_practice/config/logging.conf'
# logging.config.fileConfig(logging_conf,
#                           defaults={'fate_filename': os.path.join(log_dir, 'fate.log'),
#                                     'quant_filename': os.path.join(log_dir, 'quant.log'),
#                                     'arbiter_filename': os.path.join(log_dir, 'arbiter.log')})


# 将本地的logger注释掉，否则会覆盖掉fate的logger配置
# init_logger()
fate_logger = logging.getLogger('fate_logger')
arbiter_logger = logging.getLogger('arbiter_logger')

buf_size = 1
# 定义和实验数据记录相关的对象
sparsity_file = open(os.path.join(stats_dir, 'sparsity.csv'), 'w', buffering=buf_size)
sparsity_writer = csv.writer(sparsity_file)
sparsity_writer.writerow(['epoch', 'conv2d_sparsity', 'linear_sparsity', 'avg_sparsity'])

train_file = open(os.path.join(stats_dir, 'train.csv'), 'w', buffering=buf_size)
train_writer = csv.writer(train_file)
train_writer.writerow(['epoch', 'top1', 'top5', 'train_loss'])

valid_file = open(os.path.join(stats_dir, 'valid.csv'), 'w', buffering=buf_size)
valid_writer = csv.writer(valid_file)
valid_writer.writerow(['epoch', 'top1', 'top5', 'valid_loss'])

quant_file = open(os.path.join(stats_dir, 'quant.csv'), 'w', buffering=buf_size)
quant_writer = csv.writer(quant_file)
quant_writer.writerow(['epoch', 'top1', 'top5', 'quant_loss'])

avgloss_file = open(os.path.join(stats_dir, 'avgloss.csv'), 'w', buffering=buf_size)
avgloss_writer = csv.writer(avgloss_file)
avgloss_writer.writerow(['agg_iter', 'top1', 'top5', 'avgloss'])

client_bytes_file = open(os.path.join(stats_dir, 'client_bytes'), 'w', buffering=buf_size)
client_bytes_writer = csv.writer(client_bytes_file)
client_bytes_writer.writerow(['epoch', 'send_bytes(MB)', 'recv_bytes(MB)'])

server_bytes_file = open(os.path.join(stats_dir, 'server_bytes'), 'w', buffering=buf_size)
server_bytes_writer = csv.writer(server_bytes_file)
server_bytes_writer.writerow(['agg_iter', 'send_bytes(MB)', 'recv_bytes(MB)'])

MB = 1024 * 1024


# 设置随机数种子，使得训练结果可复现
def setup_seed(seed):
    print("正在设置随机数种子")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)


class _FedBaseContext(object):
    def __init__(self, max_num_aggregation, name):
        self.max_num_aggregation = max_num_aggregation
        self._name = name
        self._aggregation_iteration = 0
        self._early_stopped = False

    def _suffix(self, group: str = "model"):
        return (
            self._name,
            group,
            f"{self._aggregation_iteration}",
        )

    def set_stopped(self):
        self._early_stopped = True

    def increase_aggregation_iteration(self):
        self._aggregation_iteration += 1

    @property
    def aggregation_iteration(self):
        return self._aggregation_iteration

    def finished(self):
        if (
                self._early_stopped
                or self._aggregation_iteration >= self.max_num_aggregation
        ):
            return True
        return False


class FedClientContext(_FedBaseContext):
    def __init__(self, max_num_aggregation, aggregate_every_n_epoch, name="default"):
        super(FedClientContext, self).__init__(
            max_num_aggregation=max_num_aggregation, name=name
        )
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

    def send_quant_model(self, tensors, scales, weight):
        # Todo: 对p.data进行加密
        # self.aggregator.send_model(
        #     ([self.encrypt(p.data, weight) for p in tensors], weight), suffix=self._suffix(),
        # )
        # Todo: 不对p.data进行加密
        tensor_arrs = []
        # 统计发送tensor的bytes
        tensors_bytes = 0
        for tensor in tensors:
            tensor_arr = tensor.data.cpu().numpy()
            tensor_arrs.append(tensor_arr)
            tensors_bytes += tensor_arr.nbytes
        self.aggregator.send_model(
            (tensor_arrs, scales, weight), suffix=self._suffix()
        )
        return tensors_bytes

    def send_unquant_model(self, tensors, weight, compress):
        tensors_bytes = 0
        if not compress:
            tensor_arrs = []
            for tensor in tensors:
                tensor_arr = tensor.data.cpu().numpy()
                tensors_bytes += tensor_arr.nbytes
                tensor_arrs.append(tensor_arr)
            self.aggregator.send_model(
                (tensor_arrs, weight), suffix=self._suffix()
            )
        else:
            compressed_tensors, tensors_bytes = tensor_compressor.compress_tensor(tensors)
            self.aggregator.send_model(
                (compressed_tensors, weight), suffix=self._suffix()
            )
        return tensors_bytes

    def recv_model(self):
        return self.aggregator.get_aggregated_model(suffix=self._suffix())

    def send_loss(self, top1, top5, loss, weight):
        self.aggregator.send_model((top1, top5, loss, weight), suffix=self._suffix(group="loss"))

    def recv_loss(self):
        return self.aggregator.get_aggregated_model(
            suffix=self._suffix(group="convergence")
        )

    def do_quant_aggregation(self, scales, weight, device):
        send_tensors_bytes = self.send_quant_model(self._params, scales, weight)
        recv_elements: typing.Tuple[typing.List[numpy.ndarray], typing.List] = self.recv_model()
        recv_tensors_bytes = 0
        initial_agg_tensors = []
        for arr in recv_elements[0]:
            tensor = torch.from_numpy(arr)
            initial_agg_tensors.append(tensor)
            recv_tensors_bytes += arr.nbytes
        agg_tensors: typing.List[torch.Tensor] = [arr_int8.type(torch.float32).to(device)
                                                  for arr_int8 in initial_agg_tensors]
        new_scales = [scale.to(device) for scale in recv_elements[1]]

        # 对agg_tensors解量化
        scale_idx = 0
        for i in range(len(agg_tensors)):
            if len(agg_tensors[i].shape) > 1:
                q_utils.linear_dequantize(agg_tensors[i], new_scales[scale_idx], 0, inplace=True)
                scale_idx += 1
        for param, agg_tensor in zip(self._params, agg_tensors):
            if param.grad is None:
                continue
            param.data = param.data.type(torch.float32)
            param.data.copy_(agg_tensor)
        return initial_agg_tensors, send_tensors_bytes, recv_tensors_bytes

    def do_unquant_aggregation(self, weight, device, compress):
        send_tensors_bytes = self.send_unquant_model(self._params, weight, compress)
        recv_elements: typing.List = self.recv_model()
        recv_tensors_bytes = 0
        compressed = isinstance(recv_elements[0], CompressedTensor)
        if compressed:
            recv_elements, recv_tensors_bytes = tensor_compressor.restore_tensor(recv_elements)

        # 接收，拷贝
        agg_tensors = []
        for arr in recv_elements:
            if not compressed:
                recv_tensors_bytes += arr.nbytes
            agg_tensors.append(torch.from_numpy(arr).to(device))
        for param, agg_tensor in zip(self._params, agg_tensors):
            if param.grad is None:
                continue
            param.data.copy_(agg_tensor)
        return send_tensors_bytes, recv_tensors_bytes

    def do_convergence_check(self, weight, top1, top5, loss_value):
        self.loss_summary.append(loss_value)

        self.send_loss(top1, top5, loss_value, weight)

        return self.recv_loss()

    def configure_aggregation_params(
            self,
            optimizer,
    ):
        if optimizer is not None:
            self._params = [
                param
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

    def send_quant_model(self, aggregated_tensors, new_scales):
        return self.aggregator.send_aggregated_model(
            (aggregated_tensors, new_scales), suffix=self._suffix()
        )

    def send_unquant_model(self, aggregated_arrs, compressed):
        send_bytes = 0
        if compressed:
            aggregated_tensors = [torch.from_numpy(aggregated_arr) for aggregated_arr in aggregated_arrs]
            compressed_tensors, send_bytes = tensor_compressor.compress_tensor(aggregated_tensors)
            self.aggregator.send_aggregated_model(compressed_tensors, suffix=self._suffix())
        else:
            for aggregated_arr in aggregated_arrs:
                send_bytes += aggregated_arr.nbytes
            self.aggregator.send_aggregated_model(aggregated_arrs, suffix=self._suffix())
        return send_bytes

    def recv_model(self):
        return self.aggregator.get_models(suffix=self._suffix())

    def send_convergence_status(self, top1, top5, status):
        self.aggregator.send_aggregated_model(
            (top1, top5, status), suffix=self._suffix(group="convergence")
        )

    def recv_losses(self):
        return self.aggregator.get_models(suffix=self._suffix(group="loss"))

    def do_convergence_check(self):
        arbiter_logger.info('正在等待客户端发送loss')
        loss_weight_pairs = self.recv_losses()
        arbiter_logger.info('成功接受客户端发送的loss')
        total_loss = 0.0
        total_weight = 0.0
        total_top1 = 0.0
        total_top5 = 0.0

        for top1, top5, loss, weight in loss_weight_pairs:
            total_loss += loss * weight
            total_top1 += top1 * weight
            total_top5 += top1 * weight
            total_weight += weight
        mean_loss = total_loss / total_weight
        mean_top1 = total_top1 / total_weight
        mean_top5 = total_top5 / total_weight

        avgloss_writer.writerow([self.aggregation_iteration, mean_top1, mean_top5, mean_loss])

        is_converged = abs(mean_loss - self._loss) < self._eps

        self.send_convergence_status(mean_top1, mean_top5, is_converged)

        self._loss = mean_loss
        arbiter_logger.info(f'收敛性验证：loss={mean_loss},is_converged={is_converged}')
        LOGGER.info(f"convergence check: loss={mean_loss}, is_converged={is_converged}")
        return is_converged, mean_loss


class DistillerFedAggregator(object):
    def __init__(self, context: FedServerContext):
        self.context = context

    def fit(self, loss_callback):
        while not self.context.finished():
            recv_total_bytes = 0
            send_total_bytes = 0

            recv_elements: typing.List[typing.Tuple] = self.context.recv_model()

            cur_iteration = self.context.aggregation_iteration
            arbiter_logger.info(f'当前聚合轮次为{cur_iteration},成功接收到客户端的模型参数!')
            LOGGER.warn(f'收到{len(recv_elements)}客户端发送过来的模型')
            # 根据tuple中元素的个数判断是聚合后的模型还是未聚合的模型
            tuple_size = len(recv_elements[0])

            tensors = [party_tuple[0] for party_tuple in recv_elements]

            # 对tensors进行解压缩
            compressed = isinstance(tensors[0][0], CompressedTensor)
            if compressed:
                for i in range(len(tensors)):
                    tensors[i], recv_single_bytes = tensor_compressor.restore_tensor(tensors[i])
                    recv_total_bytes += recv_single_bytes

            aggregate_quant_model = tuple_size == 3
            if aggregate_quant_model:
                # 对量化后的模型进行聚合
                scales = [party_tuple[1] for party_tuple in recv_elements]
                degrees = [party_tuple[2] for party_tuple in recv_elements]
                # 对tensors进行反量化，得到float32类型的向量
                for i in range(len(tensors)):
                    tensor = tensors[i]
                    scale = scales[i]
                    scale_idx = 0
                    # Todo: 对tensor的维度进行判断，如果是
                    for j in range(len(tensor)):
                        recv_total_bytes += tensor[j].nbytes
                        if len(tensor[j].shape) > 1:
                            tensor[j] = torch.from_numpy(tensor[j].astype(np.float32))
                            q_utils.linear_dequantize(tensor[j],
                                                      scale[scale_idx], 0, inplace=True)
                            tensor[j] = tensor[j].numpy()
                            scale_idx += 1
            else:
                # 对未量化的模型进行聚合
                degrees = [party_tuple[1] for party_tuple in recv_elements]
                if not compressed:
                    for i in range(len(tensors)):
                        for j in range(len(tensors[i])):
                            recv_total_bytes += tensors[i][j].nbytes

            total_degree = sum(degrees)
            # 执行聚合过程
            for i in range(len(tensors)):
                for j, tensor in enumerate(tensors[i]):
                    tensor *= degrees[i]
                    tensor /= total_degree
                    if i != 0:
                        tensors[0][j] += tensor

            arbiter_logger.info(f'当前聚合轮次为:{cur_iteration}，聚合完成，准备向客户端分发模型')

            # Todo: 以下是发送模型的部分

            if aggregate_quant_model:
                new_scales = []
                for i in range(len(tensors[0])):
                    if len(tensors[0][i].shape) > 1:
                        tensors[0][i] = torch.from_numpy(tensors[0][i])
                        w_scale, _ = range_linear._get_quant_params_from_tensor(tensors[0][i], num_bits=8)
                        new_scales.append(w_scale)
                        range_linear.linear_quantize(tensors[0][i], w_scale, 0, inplace=True)
                        tensors[0][i] = tensors[0][i].type(torch.int8).numpy()

                        send_total_bytes += tensors[0][i].nbytes
                self.context.send_quant_model(tensors[0], new_scales)
            else:
                send_total_bytes = self.context.send_unquant_model(tensors[0], compressed)
            write_bytes(server_bytes_writer, self.context.aggregation_iteration, send_total_bytes, recv_total_bytes)
            arbiter_logger.info(f'当前聚合轮次为:{cur_iteration}，模型参数分发成功！')
            is_converged, loss = self.context.do_convergence_check()
            loss_callback(self.context.aggregation_iteration, float(loss))
            self.context.increase_aggregation_iteration()
            if is_converged:
                break

    def export_model(self, param):

        param_pb = nn_model_param_pb2.NNModelParam()

        # save api_version
        param_pb.api_version = param.api_version

        meta_pb = nn_model_meta_pb2.NNModelMeta()
        meta_pb.params.CopyFrom(param.generate_pb())
        meta_pb.aggregate_iter = self.context.aggregation_iteration

        return {_consts.MODEL_META_NAME: meta_pb, _consts.MODEL_PARAM_NAME: param_pb}

    @classmethod
    def load_model(cls, model_obj, meta_obj, param):
        param.restore_from_pb(meta_obj.params)

    @staticmethod
    def dataset_align():
        LOGGER.info("start label alignment")
        label_mapping = HomoLabelEncoderArbiter().label_alignment()
        LOGGER.info(f"label aligned, mapping: {label_mapping}")


class DistillerCompressor(object):
    def __init__(
            self,
            param,
            epochs,
            label_mapping=None,
            context: FedClientContext = None,
    ):
        self.param = copy.deepcopy(param)
        self._all_consumed_data_aggregated = True
        self.best_accuracy = 0
        self._num_data_consumed = 0
        self.context = context
        self.label_mapping = label_mapping

        (self.model, self.compression_scheduler, self.optimizer) = _init_learner(self.param, self.param.device)
        self.criterion = nn.CrossEntropyLoss().to(self.param.device)
        self.start_epoch, self.end_epoch = 0, epochs
        self.activations_collectors = create_activation_stats_collectors(self.model)
        # 这里先采用硬编码的output_dir
        # Todo: 由于guest和host执行的是同一份代码，因此，日志文件所在的目录设定为当前路径
        self.stats_dir = stats_dir

        self.tflogger = TensorBoardLogger(os.path.join(log_dir, 'tb_logs'))
        self.quant_file = os.path.join(self.stats_dir, 'acts_quantization_stats.yaml')

        self.qe_model = ...

    def get_avg_accuracy(self, top1, top5):
        return top1 * 0.5 + top5 * 0.5

    def get_label_mapping(self):
        return self.label_mapping

    def fit(self, train_loader, valid_loader):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.on_fit_epoch_start(epoch, len(train_loader.sampler))
            # 获取当前epoch下原模型的top1精度、top5精度以及损失
            top1, top5, loss = self.train_validate_with_scheduling(epoch, train_loader, valid_loader)
            fate_logger.info(f'已完成一轮训练+验证')
            LOGGER.warn(f'epoch={epoch}/{self.end_epoch},top1={top1},top5={top5},loss={loss}')
            self.on_fit_epoch_end(epoch, valid_loader, top1, top5, loss)
            if self.context.should_stop():
                break

    def on_fit_epoch_end(self, epoch, valid_loader, valid_top1, valid_top5, valid_loss):
        top1 = valid_top1
        top5 = valid_top5
        loss = valid_loss
        if self.context.should_aggregate_on_epoch(epoch):
            if self.param.quant_aware_aggregate:
                self.aggregate_quant_aware_model(epoch)
            elif self.param.post_train_quant:
                fate_logger.info('开始准备量化模型，当前epoch为{epoch}')
                self.post_train_quantize_model()
                top1, top5, loss = validate(valid_loader, self.qe_model, self.criterion, epoch, self.param.device)
                quant_writer.writerow([epoch, top1, top5, loss])

                fate_logger.info(f'量化模型验证结束,epoch = {epoch},top1 = {top1},top5 = {top5},平均loss = {loss}')
                self.aggregate_quantized_model(epoch)
            else:  # Todo: 发送原模型
                self.aggregate_unquantized_model(epoch, compress=has_pruning_policy(self.compression_scheduler))

            # Todo: 发送模型的损失，判断收敛情况
            mean_top1, mean_top5, status = self.context.do_convergence_check(
                len(valid_loader.sampler), top1, top5, loss
            )
            if status:
                self.context.set_converged()

            # 计算是否应该保存当前模型
            # avg_accuracy = self.get_avg_accuracy(mean_top1, mean_top5)
            # if avg_accuracy > self.best_accuracy:
            #     self.best_accuracy = avg_accuracy
            #     if self.param.post_train_quant:
            #         torch.save(self.qe_model.state_dict(), os.path.join(models_dir, str(epoch) + '_qe_model'))
            #     else:
            #         model_name = '_pruned_model' if has_pruning_policy(self.compression_scheduler) else '_initial_model'
            #         torch.save(self.model.state_dict(), os.path.join(models_dir, str(epoch) + '_' + model_name))
            #

            self._all_consumed_data_aggregated = True
            self._num_data_consumed = 0
            self.context.increase_aggregation_iteration()

    def train_one_epoch(self, epoch, train_loader):
        # 在此处定义收集器的上下文
        with collectors_context(self.activations_collectors['train']) as collectors:
            top1, top5, loss = train(train_loader, self.model, self.criterion, self.optimizer,
                                     epoch, self.compression_scheduler, self.param.device)
            fate_logger.info(f'训练阶段结束,epoch = {epoch},平均loss = {loss}')
            # Todo: 这里仅仅配置了稀疏水平sparsity的收集器
            # distiller.utils.log_activation_statistics(epoch, 'train', loggers=[self.tflogger],
            #                                           collector=collectors['sparsity'])

            # 获取quant_stats的收集器
            quant_save_path = os.path.join(self.stats_dir, 'acts_quantization_stats.yaml')
            quant_stats_collector = collectors['quant_stats']
            # 在前向传播结束后，需要收集统计的结果
            quant_stats_collector.save(quant_save_path)
            fate_logger.info(f'已保存量化统计数据，文件地址为：{quant_save_path}')

            # 训练结束后，记录该epoch的训练结果
            train_writer.writerow([epoch, top1, top5, loss])
        return top1, top5, loss

    def validate_one_epoch(self, epoch, valid_loader):
        top1, top5, loss = validate(valid_loader, self.model, self.criterion, epoch, self.param.device)
        fate_logger.info(f'验证阶段结束,epoch = {epoch},top1 = {top1},top5 = {top5},平均loss = {loss}')
        valid_writer.writerow([epoch, top1, top5, loss])
        return top1, top5, loss

    def train_validate_with_scheduling(self, epoch, train_loader, valid_loader):
        if self.compression_scheduler:  # Todo 这里完成剪枝的mask生成
            self.compression_scheduler.on_epoch_begin(epoch)
        learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
        LOGGER.warn(f'当前的学习率为：{learning_rate}')
        top1, top5, loss = self.train_one_epoch(epoch, train_loader)
        # # Todo: 发送未量化的模型
        # self.on_train_epoch_end(epoch)
        # 使用高精度模型在验证集上进行验证
        if valid_loader:
            top1, top5, loss = self.validate_one_epoch(epoch, valid_loader)

        if self.compression_scheduler:
            self.compression_scheduler.on_epoch_end(epoch, self.optimizer)
        return top1, top5, loss

    def summary(self):
        return {
            "loss": self.context.loss_summary,
            "is_converged": self.context.should_stop(),
        }

    def on_fit_epoch_start(self, epoch, num_samples):
        if self._all_consumed_data_aggregated:
            fate_logger.info(f'新一轮聚合开始')
            self._num_data_consumed = num_samples
            self._all_consumed_data_aggregated = False
        else:
            self._num_data_consumed += num_samples
        fate_logger.info(f'当前epoch为{epoch}，已经消耗的样本数量为{self._num_data_consumed}')

    def post_train_quantize_model(self):
        quant_args = PostQuantArgs(stats_file=self.quant_file)
        self.qe_model = copy.deepcopy(self.model)

        quantizer = quantization.PostTrainLinearQuantizer.from_args(self.qe_model, quant_args)

        # 需要dummy_input来生成计算图
        dummy_input = utils.get_dummy_input(input_shape=self.qe_model.input_shape)
        quantizer.prepare_model(dummy_input)

    def aggregate_quant_aware_model(self, epoch):
        idx = 0
        param_list = self.optimizer.param_groups[0]['params']
        first_incr = True
        scales = []
        # 仅对weight进行量化
        for module_name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                scale = module.weight_scale
                scales.append(scale.item())
                quantized_param = quantization.linear_quantize(module.float_weight,
                                                               scale,
                                                               module.weight_zero_point)
                param_list[idx].data = quantized_param.type(torch.int8)
                if first_incr:
                    idx += 2
                    first_incr = False
                else:
                    idx += 3
        # 配置optimizer中的模型参数
        self.context.configure_aggregation_params(self.optimizer)
        _, send_bytes, recv_bytes = self.context.do_quant_aggregation(scales, self._num_data_consumed,
                                                                      self.param.device)
        if self.compression_scheduler:
            self.compression_scheduler.on_aggregate_end(epoch)
        write_bytes(client_bytes_writer, epoch, send_bytes, recv_bytes)

    def aggregate_unquantized_model(self, epoch, compress):
        self.context.configure_aggregation_params(self.optimizer)
        send_bytes, recv_bytes = self.context.do_unquant_aggregation(self._num_data_consumed, self.param.device,
                                                                     compress)
        write_bytes(client_bytes_writer, epoch, send_bytes, recv_bytes)

    def aggregate_quantized_model(self, epoch):
        idx = 0
        # Todo: 参数组列表中只有一个元素吗？
        param_list = self.optimizer.param_groups[0]['params']

        first_incr = True
        # Todo: 需要仿照该方法传入量化模型的低精度参数
        for module_name, module in self.qe_model.named_modules():
            if isinstance(module, quantization.RangeLinearQuantParamLayerWrapper):
                # 权重int8
                param_list[idx].data = module.wrapped_module.weight.data.type(torch.int8)
                if first_incr:
                    idx += 2
                    first_incr = False
                else:
                    idx += 3

        # 配置optimizer中的模型参数
        self.context.configure_aggregation_params(self.optimizer)
        scales = []
        for module_name, module in self.qe_model.named_modules():
            if isinstance(module, quantization.RangeLinearQuantParamLayerWrapper):
                scales.append(module.w_scale.item())
        # 执行聚合
        initial_agg_tensors, send_bytes, recv_bytes = self.context.do_quant_aggregation(scales, self._num_data_consumed,
                                                                                        self.param.device)
        write_bytes(client_bytes_writer, epoch, send_bytes, recv_bytes)
        self.save_agg_to_qe_model(initial_agg_tensors)

    def save_agg_to_qe_model(self, initial_agg_tensors):
        idx = 0
        first_incr = True
        for module_name, module in self.qe_model.named_modules():
            if isinstance(module, quantization.RangeLinearQuantParamLayerWrapper):
                # 权重int8
                module.wrapped_module.weight.data = initial_agg_tensors[idx]

                if first_incr:
                    module.wrapped_module.bias.data = initial_agg_tensors[idx + 1]
                    idx += 2
                    first_incr = False
                else:
                    idx += 3
                module.force_readjust = torch.tensor(True)


class PostQuantArgs:
    def __init__(self, stats_file):
        self.qe_bits_acts = 8
        self.qe_bits_wts = 8
        self.qe_bits_accum = 32
        self.qe_no_quant_layers = []
        self.qe_no_clip_layers = []
        self.qe_mode = quantization.LinearQuantMode.SYMMETRIC
        self.qe_clip_acts = quantization.ClipMode.NONE
        self.qe_stats_file = stats_file


def make_dataset(data, **kwargs):
    dataset = VisionDataSet(data.path, imagenet_transform(), **kwargs)
    return dataset


# noinspection PyUnboundLocalVariable
def build_compressor(param: DistillerParam, train_data, valid_data, should_label_align=True, compressor=None):
    if compressor is None:
        epochs = param.aggregate_every_n_epoch * param.max_iter
        context = FedClientContext(
            max_num_aggregation=param.max_iter,
            aggregate_every_n_epoch=param.aggregate_every_n_epoch,
        )
        # 和服务器端进行握手
        context.init()

        expected_label_type = np.int64

        # Todo: 给data_loader加上sampler

        train_dataset = make_dataset(
            data=train_data,
            is_train=should_label_align,
            expected_label_type=expected_label_type,
        )
        valid_dataset = make_dataset(
            data=valid_data,
            is_train=False,
            expected_label_type=expected_label_type,
        )

        LOGGER.warn(f'my device is:{param.device}')
        batch_size = param.batch_size
        if batch_size < 0:
            batch_size = len(train_dataset)
        shuffle = False
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, num_workers=0,
            drop_last=True, shuffle=shuffle
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset, batch_size=batch_size, num_workers=0,
            drop_last=True, shuffle=shuffle
        )
        compressor = DistillerCompressor(param, epochs, context=context)
    return compressor, train_loader, valid_loader


def build_aggregator(param: DistillerParam, init_iteration=0):
    context = FedServerContext(
        max_num_aggregation=param.max_iter,
        eps=param.early_stop_eps
    )
    context.init(init_aggregation_iteration=init_iteration)
    fed_aggregator = DistillerFedAggregator(context)
    return fed_aggregator


def imagenet_transform():
    resize, crop = 256, 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return transform


def _init_learner(param, device='cpu'):
    model = create_model(param.pretrained, param.dataset, param.arch, num_classes=param.num_classes, device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=param.lr)

    compression_scheduler = None
    # 如果有配置sched_dict的话
    if param.sched_dict:
        compression_scheduler = config_scheduler(model, optimizer, param.sched_dict, compression_scheduler)
    return model, compression_scheduler, optimizer


def train(train_loader, model, criterion, optimizer, epoch, compression_scheduler, device):
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    # 记录一个epoch训练的日志信息
    fate_logger.info(f'开始一轮训练，epoch为:{epoch}，batch_size为:{batch_size}，每个epoch需要的step为:{steps_per_epoch}')
    model.train()
    OVERALL_LOSS_KEY = 'Overall Loss'
    OBJECTIVE_LOSS_KEY = 'Objective Loss'
    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    LOGGER.warn(f'训练总样本数为：{total_samples}')

    for train_step, (inputs, target) in enumerate(train_loader):
        inputs = inputs.to(device)
        target = target.to(device)
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

        output = model(inputs)

        loss = criterion(output, target)
        classerr.add(output.detach(), target)
        losses[OBJECTIVE_LOSS_KEY].add(loss.item())
        fate_logger.info(f'\tepoch = {epoch}，train_step = {train_step},loss = {loss.item()}')
        # Fate的logger
        LOGGER.warn(
            f'\tepoch = {epoch}，train_step = {train_step}/{steps_per_epoch},top1 = {classerr.value(1)},top5 = {classerr.value(5)}, loss = {loss.item()}')

        if compression_scheduler:
            agg_loss = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss,
                                                                  optimizer=optimizer, return_loss_components=True)
            loss = agg_loss.overall_loss
            losses[OVERALL_LOSS_KEY].add(loss.item())
            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                losses[lc.name].add(lc.value.item())
        else:
            losses[OVERALL_LOSS_KEY].add(loss.item())
        optimizer.zero_grad()
        loss.backward()
        if compression_scheduler:
            compression_scheduler.before_parameter_optimization(epoch, train_step, steps_per_epoch, optimizer)
        optimizer.step()

        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)
    # 当前epoch结束后，记录模型的稀疏水平
    record_sparsity(epoch, model)
    return classerr.value(1), classerr.value(5), losses[OBJECTIVE_LOSS_KEY].mean


def validate(valid_loader, model, criterion, epoch, device):
    objective_loss = tnt.AverageValueMeter()
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    total_samples = len(valid_loader.sampler)
    batch_size = valid_loader.batch_size
    # 保留该变量用于输出验证进度
    total_steps = math.ceil(total_samples / batch_size)

    fate_logger.info(f'开始一轮验证，epoch为:{epoch}，batch_size为:{batch_size}，每个epoch需要的step为:{total_steps}')

    # 切换到评估模式
    model.eval()
    with torch.no_grad():
        for validate_step, (inputs, target) in enumerate(valid_loader):
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output, target)
            objective_loss.add(loss.item())

            fate_logger.info(f'\tepoch = {epoch}，valid_step = {validate_step},loss = {loss.item()}')
            classerr.add(output.detach(), target)
            LOGGER.warn(
                f'\tepoch = {epoch}，valid_step = {validate_step}/{total_steps}, top1 = {classerr.value(1)}, top5 = {classerr.value(5)}, loss = {loss.item()}')

    return classerr.value(1), classerr.value(5), objective_loss.mean


def create_activation_stats_collectors(model):
    # Todo: 这里先将一些参数硬编码
    # 先只在test阶段进行指标统计任务
    phases = ['train', 'valid', 'test']

    class MissingDict(dict):
        # 当key不存在的时候，调用missing魔术方法，而不抛出KeyError错误
        def __missing__(self, key):
            return None

    genCollectors = lambda: MissingDict({
        "sparsity": SummaryActivationStatsCollector(model, "sparsity",
                                                    # 传入一个匿名函数，即collector定义中的test_fn
                                                    lambda t: 100 * distiller.utils.sparsity(t)),
        "l1_channels": SummaryActivationStatsCollector(model, "l1_channels",
                                                       distiller.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         distiller.utils.activation_channels_apoz),
        "mean_channels": SummaryActivationStatsCollector(model, "mean_channels",
                                                         distiller.utils.activation_channels_means),
        "records": RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d]),
        "quant_stats": QuantCalibrationStatsCollector(model)
    })
    # 根据传入的阶段信息，返回一个字典阶段到收集器集合的信息
    return {k: (genCollectors() if k in phases else MissingDict())
            for k in ('train', 'valid', 'test')}


def record_sparsity(epoch, model):
    # 加权的稀疏水平
    TOTAL_NUM, ZERO_NUM = 'total_num', 'zero_num'
    classes = (torch.nn.Conv2d, torch.nn.Linear)

    sparsities = {}
    for clz in classes:
        sparsities[clz] = {TOTAL_NUM: 0, ZERO_NUM: 0}
    for module_name, module in model.named_modules():
        if isinstance(module, classes):
            zero_num, total_num = utils.zero_total_num(module.weight)
            clz = type(module)
            sparsities[clz][TOTAL_NUM] += total_num
            sparsities[clz][ZERO_NUM] += zero_num
    # 计算稀疏水平
    total_zero = 0
    total_num = 0
    s_list = [epoch]
    for clz, info in sparsities.items():
        s_list.append(info[ZERO_NUM] / info[TOTAL_NUM])
        total_zero += info[ZERO_NUM]
        total_num += info[TOTAL_NUM]
    s_list.append(total_zero / total_num)
    sparsity_writer.writerow(s_list)


def has_pruning_policy(compression_scheduler):
    for policy in compression_scheduler.sched_metadata:
        if isinstance(policy, PruningPolicy):
            return True
    return False


def write_bytes(writer, epoch, send_bytes, recv_bytes):
    writer.writerow([epoch, float(format(send_bytes / MB, '.2f')), float(format(recv_bytes / MB, '.2f'))])
