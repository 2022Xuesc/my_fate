import contextlib
from functools import partial, reduce
import operator
import enum

import xlsxwriter as xlsxwriter
import yaml
import os
from sys import float_info
from collections import OrderedDict
from contextlib import contextmanager
import torch
from torchnet.meter import AverageValueMeter
import logging
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt

import federatedml.nn.backend.distiller as distiller

import federatedml.nn.backend.distiller.utils

import numpy as np
import concurrent.futures

matplotlib.use('Agg')

msglogger = logging.getLogger()

__all__ = ['SummaryActivationStatsCollector', 'RecordsActivationStatsCollector', 'QuantCalibrationStatsCollector',
           'ActivationHistogramsCollector', 'RawActivationsCollector', 'CollectorDirection',
           'collect_quant_stats', 'collect_histograms', 'collect_raw_outputs',
           'collector_context', 'collectors_context']


# 定义收集器的方向类，属于枚举类型
# Todo: 这些方向的标志
class CollectorDirection(enum.Enum):
    OUT = 0
    # 输出Feature Map
    OFM = 0
    IN = 1
    # 输入Feature Map
    IFM = 1
    IFMS = 1


class ActivationStatsCollector(object):
    """ 收集模型激活单元的统计信息

    该类是收集激活单元统计数据的基类
    收集了优化过程中不同阶段的统计数据，统计数据通过.value()或者独立的模块进行访问

    当前的实现有以下几个说明：
    1. 配置后十分缓慢
    2. 无法访问torch.Functions的激活单元，只能访问torch.Modules的激活单元

    该收集器使用模型的前向钩子来访问feature-maps,这是缓慢的，并且限制我们只能得到torch.Modules的输出。
    可以通过classes参数选择只观察特定的层或者时在验证和测试阶段使用该收集器，来加快整个优化过程。
    """

    def __init__(self, model, stat_name, classes):
        """
        Args:
            model - 需要检测的模型
            stat_name - 需要收集的统计信息的名称/类型
                通过module.<stat_name>来访问激活单元的统计数据，例如module.sparsity
            classes - 一个收集统计信息的层的列表：使用空列表或者None将采集所有层的统计信息
        """
        super(ActivationStatsCollector, self).__init__()
        self.model = model
        self.stat_name = stat_name
        self.classes = classes
        # 定义前向传播时的钩子函数
        self.fwd_hook_handles = []

        # torch.Modules没有网络层的名称，因此，需要为每层声明一个可读的名称
        distiller.utils.assign_layer_fq_names(model)

        self._dont_collect_list = []
        # Currently, this is internal, and its only purpose is to enable skipping collection
        # for wrapped modules inside post-training quantization wrapper classes.
        # When doing PTQ, the outputs of these wrapped modules are actually intermediate results
        # which are not relevant for tracking.
        # 跳过量化的类
        # self._dont_collect_list = [module.wrapped_module.distiller_name for module in model.modules() if
        #                            is_post_train_quant_wrapper(module)]

    def value(self):
        """返回包含{layer_name: statistic}的字典"""
        activation_stats = OrderedDict()
        # 具体逻辑由其子类实现
        # 为模块的每一个子模块都执行_collect操作,并且传入字典的引用-->inplace
        self.model.apply(partial(self._collect_activations_stats, activation_stats=activation_stats))
        return activation_stats

    def start(self, modules_list=None):
        """开始收集激活单元的统计信息

        This will iteratively register the modules' forward-hooks, so that the collector
        will be called from the forward traversal and get exposed to activation data.
        modules_list (iterable): track stats for modules in the list. If None/empty - will track for all modules.
        """
        assert len(self.fwd_hook_handles) == 0
        if not modules_list:
            # apply方法将self.start_module递归地应用于模块中的每个子模块及其自身，典型的用法是对一个model的参数进行初始化
            # Todo: 这里的是注册前向钩子
            self.model.apply(self.start_module)
            return
        # 将named_modules转换成dict形式，便于通过层名称进行索引
        modules_dict = dict(self.model.named_modules())
        for module_name in modules_list:
            modules_dict[module_name].apply(self.start_module)

    def start_module(self, module):
        """为所有符合条件的模块迭代地注册前向钩子函数

        Eligible modules are currently filtered by their class type.
        """
        if self._should_collect(module):
            # 给激活单元的回调注册前向钩子
            self.fwd_hook_handles.append(module.register_forward_hook(self._activation_stats_cb))
            # 开始计数器
            self._start_counter(module)

    def stop(self):
        """停止收集统计数据

        This will iteratively unregister the modules' forward-hooks.
        """
        # 移除前向钩子
        for handle in self.fwd_hook_handles:
            handle.remove()
        self.fwd_hook_handles = []

    def reset(self):
        """Reset the statistics counters of this collector."""
        # 重置计数器
        self.model.apply(self._reset_counter)
        return self

    def save(self, fname):
        raise NotImplementedError

    def _activation_stats_cb(self, module, inputs, output):
        """Handle new activations ('output' argument).

        当module模块前向传播时，激活该回调函数
        """
        raise NotImplementedError

    def _start_counter(self, module):
        """Start a specific statistic counter - this is subclass-specific code"""
        raise NotImplementedError

    def _reset_counter(self, module):
        """Reset a specific statistic counter - this is subclass-specific code"""
        raise NotImplementedError

    def _collect_activations_stats(self, module, activation_stats, name=''):
        """Handle new activations - this is subclass-specific code"""
        raise NotImplementedError

    def _should_collect(self, module):
        """判定给定的模块是否需要收集统计信息

        :param module:
        :return:
        """
        if module.distiller_name in self._dont_collect_list:
            return False
        # 仅为叶子模块收集统计信息
        # We make an exception for models that were quantized with 'PostTrainLinearQuantizer'. In these
        # models, the quantized modules are actually wrappers of the original FP32 modules, so they are
        # NOT leaf modules - but we still want to track them.

        # 如果该模块有子模块（非叶子节点）并且不属于post-train_quant_wrapper等，不进行性能追踪
        # if distiller.has_children(module) and not (is_post_train_quant_wrapper(module) or
        #                                            isinstance(module, QFunctionalWrapper)):
        #     return False

        # Identity模块通常是建立一个输入模块，什么都不做。
        if isinstance(module, torch.nn.Identity):
            return False

        register_all_class_types = not self.classes
        if register_all_class_types or isinstance(module, tuple(self.classes)):
            return True

        return False


class WeightedAverageValueMeter(AverageValueMeter):
    """
    对torchnet的平均度量的纠正，后者在实现标准差收集时没有考虑批量大小
    """

    def __init__(self):
        super().__init__()
        self.std = None
        self.var = None
        self.mean_old = None
        self.mean = None
        self.m_s = None

    def add(self, value, n=1):
        self.sum += value * n
        if n <= 0:
            raise ValueError("Cannot use a non-positive weight for the running stat.")
        elif self.n == 0:
            self.mean = 0.0 + value  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + n * (value - self.mean_old) / float(self.n + n)
            self.m_s += n * (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n + n - 1.0))
        self.var = self.std ** 2

        self.n += n


class SummaryActivationStatsCollector(ActivationStatsCollector):
    """该类收集激活单元统计数据的总结
     计算激活单元统计数据的均值，它相比于收集每个激活单元的记录来说十分轻量，且很快
     统计函数在构造器中进行配置
     inputs_consolidate_func在张量元组上调用，且返回一个张量。
    """

    # Todo: classes为什么只有Relu相关的？
    # Answer: 这里的默认是relu，可以自己配置，比如conv2d
    def __init__(self, model, stat_name, summary_fn,
                 classes=(torch.nn.ReLU, torch.nn.ReLU6, torch.nn.LeakyReLU),
                 collector_direction=CollectorDirection.OUT,
                 inputs_consolidate_func=torch.cat):
        super(SummaryActivationStatsCollector, self).__init__(model, stat_name, classes)
        self.summary_fn = summary_fn
        self.collector_direction = collector_direction
        self.inputs_func = inputs_consolidate_func

    def _activation_stats_cb(self, module, inputs, output):
        """记录模块的激活单元稀疏度

        这是模块forward()函数的一个回调
        """
        # 根据收集器的方向指定feature_map
        feature_map = output if self.collector_direction == CollectorDirection.OUT else self.inputs_func(inputs)
        try:
            # 对feature-map进行统计，并将统计信息添加到模型的stat_name域中
            # Todo: 这里的属性不需要先设定吗？
            # Answer： 是在_start_counter()方法中设定的
            getattr(module, self.stat_name).add(self.summary_fn(feature_map.data), feature_map.data.numel())
        except RuntimeError as e:
            if "The expanded size of the tensor" in e.args[0]:
                # Todo: 参考prepare a module for quantization
                raise ValueError(
                    "ActivationStatsCollector: a module ({} - {}) was encountered twice during model.apply().\n"
                    "This is an indication that your model is using the same module instance, "
                    "in multiple nodes in the graph.  This usually occurs with ReLU modules: \n"
                    "For example in TorchVision's ResNet model, self.relu = nn.ReLU(inplace=True) is "
                    "instantiated once, but used multiple times.  This is not permissible when using "
                    "instances of ActivationStatsCollector.".
                        format(module.distiller_name, type(module)))
            else:
                msglogger.info("Exception in _activation_stats_cb: {} {}".format(module.distiller_name, type(module)))
                raise

    # 执行初始化操作，比如设定名称，度量对象等
    def _start_counter(self, module):
        if not hasattr(module, self.stat_name):
            # 给module设置stat_name属性，初始化度量对象为权重平均度量，对于每层的求平均等指标
            setattr(module, self.stat_name, WeightedAverageValueMeter())
            # 给summary赋上一个名称
            if hasattr(module, 'distiller_name'):
                getattr(module, self.stat_name).name = module.distiller_name
            else:
                getattr(module, self.stat_name).name = '_'.join((
                    module.__class__.__name__, str(id(module))))

    def _reset_counter(self, module):
        if hasattr(module, self.stat_name):
            # 重置stat_name属性
            getattr(module, self.stat_name).reset()

    def _collect_activations_stats(self, module, activation_stats, name=''):
        # 从模块中取出统计数据，然后将其记录到activation_stats中
        if hasattr(module, self.stat_name):
            mean = getattr(module, self.stat_name).mean
            if isinstance(mean, torch.Tensor):
                mean = mean.tolist()
            activation_stats[getattr(module, self.stat_name).name] = mean

    # Todo: 本地化操作之后再看，先搞清楚统计的逻辑
    def save(self, fname):
        """将统计结果保存到excel workbook中

        :param fname: 文件名称
        :return:
        """
        if not fname.endswith('.xlsx'):
            fname = '.'.join([fname, 'xlsx'])
        with contextlib.suppress(OSError):
            os.remove(fname)

        # noinspection PyShadowingNames
        def _add_worksheet(workbook, tab_name, record):
            try:
                worksheet = workbook.add_worksheet(tab_name)
            except xlsxwriter.exceptions.InvalidWorksheetName:
                worksheet = workbook.add_worksheet()

            col_names = []
            for col, (module_name, module_summary_data) in enumerate(record.items()):
                if not isinstance(module_summary_data, list):
                    module_summary_data = [module_summary_data]
                worksheet.write_column(1, col, module_summary_data)
                col_names.append(module_name)
            worksheet.write_row(0, 0, col_names)

        with xlsxwriter.Workbook(fname) as workbook:
            _add_worksheet(workbook, self.stat_name, self.value())

        return fname


class RecordsActivationStatsCollector(ActivationStatsCollector):
    """这个类收集激活单元统计结果的记录

    计算激活单元统计数据的一个硬编码的集合，然后为每个激活单元收集记录。
    整个模型的激活单元记录可被保存到Excel workbook中

    For obvious reasons, this is slower than SummaryActivationStatsCollector.
    """

    def __init__(self, model, classes=(torch.nn.ReLU,
                                       torch.nn.ReLU6,
                                       torch.nn.LeakyReLU)):
        super(RecordsActivationStatsCollector, self).__init__(model, "statistics_records", classes)

    # Todo: 该方法结合具体的实例看，主要关注维度的处理
    def _activation_stats_cb(self, module, inputs, output):
        """记录模块中激活单元的稀疏水平

        This is a callback from the forward() of 'module'.
        """

        def to_np(stats):
            if isinstance(stats, tuple):
                return stats[0].detach().cpu().numpy()
            else:
                return stats.detach().cpu().numpy()

        # 从一个激活单元的批次中收集统计数据
        if not output.is_contiguous():
            output = output.contiguous()
        # 将输出展平，output.size(0)是batch_size
        act = output.view(output.size(0), -1)
        # 计算批次中的最小、最大以及均值
        batch_min_list = to_np(torch.min(act, dim=1)).tolist()
        batch_max_list = to_np(torch.max(act, dim=1)).tolist()
        batch_mean_list = to_np(torch.mean(act, dim=1)).tolist()
        # 如果激活单元仅包含一个元素，则标准差是无意义的，因此，返回0
        if act.shape[0] == act.numel():
            batch_std_list = to_np(torch.zeros(act.shape[0])).tolist()
        else:
            batch_std_list = to_np(torch.std(act, dim=1)).tolist()
        # 计算l2范数
        batch_l2_list = to_np(torch.norm(act, p=2, dim=1)).tolist()

        # 这个字典在start_counter时创建
        module.statistics_records['min'].extend(batch_min_list)
        module.statistics_records['max'].extend(batch_max_list)
        module.statistics_records['mean'].extend(batch_mean_list)
        module.statistics_records['std'].extend(batch_std_list)
        module.statistics_records['l2'].extend(batch_l2_list)
        module.statistics_records['shape'] = distiller.utils.size2str(output)

    @staticmethod
    def _create_records_dict():
        """创建记录的字典，包含min、max、mean、std、l2、shape字段

        :return: 返回创建好的字典实例
        """
        records = OrderedDict()
        for stat_name in ['min', 'max', 'mean', 'std', 'l2']:
            records[stat_name] = []
        records['shape'] = ''
        return records

    # Todo: 保存到本地，之后再看
    def save(self, fname):
        """Save the records to an Excel workbook, with one worksheet per layer.
        """
        fname = ".".join([fname, 'xlsx'])
        try:
            os.remove(fname)
        except OSError:
            pass

        records_dict = self.value()
        with xlsxwriter.Workbook(fname) as workbook:
            for module_name, module_act_records in records_dict.items():
                try:
                    worksheet = workbook.add_worksheet(module_name)
                except xlsxwriter.exceptions.InvalidWorksheetName:
                    worksheet = workbook.add_worksheet()

                col_names = []
                for col, (col_name, col_data) in enumerate(module_act_records.items()):
                    if col_name == 'shape':
                        continue
                    worksheet.write_column(1, col, col_data)
                    col_names.append(col_name)
                worksheet.write_row(0, 0, col_names)
                worksheet.write(0, len(col_names) + 2, module_act_records['shape'])
        return fname

    def _start_counter(self, module):
        """
        为模块初始化记录的属性
        :param module: 需要统计的模块
        :return:
        """
        if not hasattr(module, "statistics_records"):
            module.statistics_records = self._create_records_dict()

    # 重新设置记录字典
    def _reset_counter(self, module):
        if hasattr(module, "statistics_records"):
            module.statistics_records = self._create_records_dict()

    # 将记录收集到activation_stats中
    # Todo: 在调用.value()方法时赋值
    def _collect_activations_stats(self, module, activation_stats, name=''):
        if hasattr(module, "statistics_records"):
            activation_stats[module.distiller_name] = module.statistics_records


# 暂时不支持数据并行模块
def _verify_no_dataparallel(model):
    if torch.nn.DataParallel in [type(m) for m in model.modules()]:
        raise ValueError('Model contains DataParallel modules, which can cause inaccurate stats collection. '
                         'Either create a model without DataParallel modules, or call '
                         'distiller.utils.make_non_parallel_copy on the model before invoking the collector')


class _QuantStatsRecord(object):
    """
    量化的统计数据记录
    """

    @staticmethod
    def create_records_dict():
        records = OrderedDict()
        records['min'] = float_info.max
        records['max'] = -float_info.max
        for stat_name in ['avg_min', 'avg_max', 'mean', 'std', 'b']:
            records[stat_name] = 0
        records['shape'] = ''
        records['total_numel'] = 0
        return records

    def __init__(self):
        # We don't know the number of inputs at this stage, so we defer records creation to the actual callback
        # 由于不知道该阶段输入的数目，因此，将记录的创建推迟到实际的回调函数中
        self.inputs = []
        self.output = self.create_records_dict()


# Todo: 配置好量化后再细看该统计类
class QuantCalibrationStatsCollector(ActivationStatsCollector):
    """该类对量化需要的激活单元统计信息进行追踪，对于每一层的输入和输出
     The tracked stats are:
      * Absolute min / max
      * Average min / max (calculate min / max per sample and average those)
      * Overall mean
      * Overall standard-deviation

    The generated stats dict has the following structure per-layer:
    'layer_name':
        'inputs':
            0:
                'min': value
                'max': value
                ...
            ...
            n:
                'min': value
                'max': value
                ...
        'output':
            'min': value
            'max': value
            ...
    Where n is the number of inputs the layer has.
    The calculated stats can be saved to a YAML file.

    If a certain layer operates in-place, that layer's input stats will be overwritten by its output stats.
    The collector can, optionally, check for such cases at runtime. In addition, a simple mechanism to disable inplace
    operations in the model can be used. See arguments details below.

    Args:
        model (torch.nn.Module): The model we are monitoring
        classes (list): List of class types for which we collect activation statistics. Passing an empty list or
          None will collect statistics for all class types.
        inplace_runtime_check (bool): If True will raise an error if an in-place operation is detected
        disable_inplace_attrs (bool): If True, will search all modules within the model for attributes controlling
          in-place operations and disable them.
        inplace_attr_names (iterable): If disable_inplace_attrs is enabled, this is the list of attribute name
          that will be searched for.

    TODO: Consider merging with RecordsActivationStatsCollector
    Current differences between the classes:
      * Track single value per-input/output-per-module for the entire run. Specifically, for standard deviation this
        cannot be done by tracking per-activation std followed by some post-processing
      * Track inputs in addition to outputs
      * Different serialization (yaml vs xlsx)
    """

    def __init__(self, model, classes=None, inplace_runtime_check=False,
                 disable_inplace_attrs=False, inplace_attr_names=('inplace',)):
        super(QuantCalibrationStatsCollector, self).__init__(model, "quant_stats", classes)

        _verify_no_dataparallel(model)

        self.batch_idx = 0
        self.inplace_runtime_check = inplace_runtime_check
        self.collecting_second_pass = False

        if disable_inplace_attrs:
            if not inplace_attr_names:
                raise ValueError('inplace_attr_names cannot by empty or None')
            for m in model.modules():
                for n in inplace_attr_names:
                    if hasattr(m, n):
                        setattr(m, n, False)

    # 检查统计数据的一些要求
    def _check_required_stats(self):
        """
        Check whether the required statistics were collected to allow collecting laplace distribution stats.
        """
        for name, module in self.model.named_modules():
            if not self._should_collect(module):
                continue
            if not hasattr(module, 'quant_stats'):
                raise RuntimeError('Collection of Laplace distribution statistics is '
                                   'only allowed after collection of stats has started.')
            for i, input_stats_record in enumerate(module.quant_stats.inputs):
                if 'mean' not in input_stats_record:
                    raise RuntimeError('The required stats for input[%d] in module "%s" were not collected. '
                                       'Please collect the required statistics using `collector.start()` and evaluating'
                                       ' the model for enough batches.' % (i, name))
            if 'mean' not in module.quant_stats.output:
                raise RuntimeError('The required stats for the output in module "%s" were not collected. '
                                   'Please collect the required statistics using `collector.start()` and evaluating'
                                   ' the model for enough batches.' % name)

    def start_second_pass(self):
        self._check_required_stats()
        self.collecting_second_pass = True
        # 为所有追踪的模块重置batch_idx
        for module in self.model.modules():
            if not self._should_collect(module):
                continue
            module.batch_idx = 0
            for record in module.quant_stats.inputs:
                record['total_numel'] = 0
            module.quant_stats.output['total_numel'] = 0

    def stop_second_pass(self):
        self.collecting_second_pass = False

    def _activation_stats_cb(self, module, inputs, output):
        """
        A callback for updating the required statistics for quantization in a module.
        """

        # 更新运行时的均值
        def update_running_mean(values, prev_mean, total_values_so_far):
            """
            Updates a running mean of a tensor of values
            Args:
                values (torch.Tensor): the new tensor
                prev_mean (float): the previous running mean
                total_values_so_far (int): the number of the values so far
            """
            curr_numel = values.numel()
            prev_numel = total_values_so_far
            return (prev_numel * prev_mean + values.sum().item()) / (prev_numel + curr_numel)

        # 更新标准差
        def update_std(values, prev_std, mean, total_values_so_far):
            """
            Updates std of the tensor
            """
            prev_variance = prev_std ** 2
            curr_sqr_dists = (values - mean) ** 2
            new_variance = update_running_mean(curr_sqr_dists, prev_variance, total_values_so_far)
            return sqrt(new_variance)

        # 更新拉普拉斯分布的b参数值
        def update_b(values, previous_b, mean, total_values_so_far):
            """
            Updates the 'b' parameter of Laplace Distribution.
            """
            curr_abs_dists = (values - mean).abs_()
            return update_running_mean(curr_abs_dists, previous_b, total_values_so_far)

        # 更新张量的记录
        def update_record(record, tensor):
            if tensor.dtype not in [torch.float16, torch.float32, torch.float64]:
                # Mean function only works for float tensors
                tensor = tensor.to(torch.float32)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            act = tensor.view(tensor.size(0), -1)
            numel = act.numel()
            if self.collecting_second_pass:
                record['b'] = update_b(act, record['b'], record['mean'], record['total_numel'])
                record['std'] = update_std(act, record['std'], record['mean'], record['total_numel'])
                record['total_numel'] += numel
                return

            # In the general case, the average min/max that we're collecting are averages over the per-sample
            # min/max values. That is - we first calculate the min/max for each sample in the batch, then average
            # over that.
            # But - If each sample contains just a single value, then such a per-sample calculation we'll result in
            # avg_min = avg_max. So in that case we "revert" to calculating "global" values, for the whole batch,
            # instead of per-sample values
            dim = 0 if numel == act.shape[0] else 1

            min_per_sample = act.min(dim=dim)[0]
            max_per_sample = act.max(dim=dim)[0]
            record['min'] = min(record['min'], min_per_sample.min().item())
            record['max'] = max(record['max'], max_per_sample.max().item())
            record['avg_min'] = update_running_mean(min_per_sample, record['avg_min'], record['total_numel'])
            record['avg_max'] = update_running_mean(max_per_sample, record['avg_max'], record['total_numel'])
            new_mean = update_running_mean(act, record['mean'], record['total_numel'])
            record['mean'] = new_mean
            record['total_numel'] += numel

            if not record['shape']:
                record['shape'] = distiller.utils.size2str(tensor)

        if self.inplace_runtime_check and any([id(input) == id(output) for input in inputs]):
            # noinspection PyProtectedMember
            if not isinstance(module, torch.nn.modules.dropout._DropoutNd):
                raise RuntimeError('Inplace operation detected, meaning inputs stats are overridden by output stats. '
                                   'You can either disable this check or make sure no in-place operations occur. '
                                   'See QuantCalibrationStatsCollector class documentation for more info.')

        module.batch_idx += 1

        if not module.quant_stats.inputs:
            # 推迟输入记录的初始化，因为当前我们只知道输入的数量
            for i in range(len(inputs)):
                module.quant_stats.inputs.append(_QuantStatsRecord.create_records_dict())

        with torch.no_grad():
            for idx, input in enumerate(inputs):
                update_record(module.quant_stats.inputs[idx], input)
            update_record(module.quant_stats.output, output)

    def _start_counter(self, module):
        # 该函数主要进行设置属性等操作
        # 在该阶段不知道输入的数量，因此，将记录的创建推迟到回调函数执行时
        module.quant_stats = _QuantStatsRecord()
        module.batch_idx = 0

    def _reset_counter(self, module):
        # Todo: 推迟
        if hasattr(module, 'quant_stats'):
            module.quant_stats = _QuantStatsRecord()
            module.batch_idx = 0

    def _collect_activations_stats(self, module, activation_stats, name=''):
        if not hasattr(module, 'quant_stats'):
            return
        # 设置模型量化的输入和输出
        activation_stats[module.distiller_name] = OrderedDict()
        if module.quant_stats.inputs:
            activation_stats[module.distiller_name]['inputs'] = OrderedDict()
            for idx, sr in enumerate(module.quant_stats.inputs):
                activation_stats[module.distiller_name]['inputs'][idx] = sr
        activation_stats[module.distiller_name]['output'] = module.quant_stats.output

    # 将收集结果保存到yaml文件中
    def save(self, fname):
        if not fname.endswith('.yaml'):
            fname = ".".join([fname, 'yaml'])
        try:
            os.remove(fname)
        except OSError:
            pass

        records_dict = self.value()
        distiller.utils.yaml_ordered_save(fname, records_dict)

        return fname


class ActivationHistogramsCollector(ActivationStatsCollector):
    """该类为每一层的每个输入和输出张量收集激活单元的直方图
        1. 他需要预计算好的每个张量的min/max统计信息，防止在运行期间存储所有激活单元的需要。
        2. 根据这些min/max值创建直方图，并且在每个迭代后更新
        3. 任何超出预计算范围内的值将会被钳制

    # Todo: 这里统计堆bin的含义是什么
    # 分箱数，调用该函数时，指定将给定的list分成多少个段，然后函数给出每个分段样本的个数。

    The generated stats dict has the following structure per-layer:
    'layer_name':
        'inputs':
            0:
                'hist': tensor             # Tensor with bin counts
                'bin_centroids': tensor    # Tensor with activation values corresponding to center of each bin
            ...
            n:
                'hist': tensor
                'bin_centroids': tensor
        'output':
            'hist': tensor
            'bin_centroids': tensor
    Where n is the number of inputs the layer has.
    The generated stats dictionary can be saved to a file.
    Optionally, histogram images for all tensor can be saved as well

    Args:
        model (torch.nn.Module): The model we are monitoring
        activation_stats (str / dict): Either a path to activation stats YAML file, or a dictionary containing
          the stats. The stats are expected to be in the same structure as generated by QuantCalibrationStatsCollector.
        classes (list): List of class types for which we collect activation statistics. Passing an empty list or
          None will collect statistics for all class types.
        nbins (int): Number of histogram bins，均分
        save_hist_imgs (bool): If set, calling save() will dump images of the histogram plots in addition to saving the
          stats dictionary
        hist_imgs_ext (str): The file type to be used when saving histogram images
    """

    def __init__(self, model, activation_stats, classes=None, nbins=2048,
                 save_hist_imgs=False, hist_imgs_ext='.svg'):
        super(ActivationHistogramsCollector, self).__init__(model, 'histogram', classes)

        _verify_no_dataparallel(model)

        # 需要预计算的统计信息文件，将其读取成dict
        if isinstance(activation_stats, str):
            if not os.path.isfile(activation_stats):
                raise ValueError("Model activation stats file not found at: " + activation_stats)
            msglogger.info('Loading activation stats from: ' + activation_stats)
            with open(activation_stats, 'r') as stream:
                activation_stats = distiller.utils.yaml_ordered_load(stream)
        elif not isinstance(activation_stats, (dict, OrderedDict)):
            raise TypeError('model_activation_stats must either be a string, a dict / OrderedDict or None')

        self.act_stats = activation_stats
        self.nbins = nbins
        self.save_imgs = save_hist_imgs
        self.imgs_ext = hist_imgs_ext if hist_imgs_ext[0] == '.' else '.' + hist_imgs_ext

    def _get_min_max(self, *keys):
        # 使用reduce函数，获取最小值和最大值。
        # keys是一些主键，相当于统计的dim。对指定主键下的所有的值执行reduce操作
        stats_entry = reduce(operator.getitem, keys, self.act_stats)
        return stats_entry['min'], stats_entry['max']

    # 定义前向传播时的回调函数
    # Todo: apply时，会传入相关的参数
    def _activation_stats_cb(self, module, inputs, output):
        # noinspection PyShadowingNames
        def get_hist(t, stat_min, stat_max):
            # torch.histc 无法在整型数据类型上生效，因此，对其进行转换
            if t.dtype not in [torch.float, torch.double, torch.half]:
                t = t.float()
            # 对传入的值进行钳制
            t_clamped = t.clamp(stat_min, stat_max)
            # 计算张量的直方图，元素被分类成min和max之间相等宽度的单元格
            hist = torch.histc(t_clamped.cpu(), bins=self.nbins, min=stat_min, max=stat_max)
            return hist

        with torch.no_grad():
            # 对每个输入计算直方图，并分样本维度上进行累加
            for idx, input in enumerate(inputs):
                # Todo: 钳制到最小值到最大值之间不是histc已经实现了吗？
                stat_min, stat_max = self._get_min_max(module.distiller_name, 'inputs', idx)
                curr_hist = get_hist(input, stat_min, stat_max)
                # Todo: 为什么使用加？层之间的统计数据进行累加吗？层之间结构不同如何累加？
                # Answer: Module是一个层
                module.input_hists[idx] += curr_hist

            # 计算输出的直方图
            stat_min, stat_max = self._get_min_max(module.distiller_name, 'output')
            curr_hist = get_hist(output, stat_min, stat_max)
            module.output_hist += curr_hist

    def _reset(self, module):
        num_inputs = len(self.act_stats[module.distiller_name]['inputs'])
        # Todo: 这里为什么连续赋值两次？
        module.input_hists = module.input_hists = [torch.zeros(self.nbins) for _ in range(num_inputs)]
        module.output_hist = torch.zeros(self.nbins)

    def _start_counter(self, module):
        self._reset(module)

    def _reset_counter(self, module):
        if hasattr(module, 'output_hist'):
            self._reset(module)

    def _collect_activations_stats(self, module, activation_stats, name=''):
        if not hasattr(module, 'output_hist'):
            return

        # 返回直方图数据以及分箱中心点
        # noinspection PyShadowingNames
        def get_hist_entry(min_val, max_val, hist):
            od = OrderedDict()
            od['hist'] = hist
            bin_width = (max_val - min_val) / self.nbins
            # 计算分箱的中心点
            od['bin_centroids'] = torch.linspace(min_val + bin_width / 2, max_val - bin_width / 2, self.nbins)
            return od

        stats_od = OrderedDict()
        inputs_od = OrderedDict()
        for idx, hist in enumerate(module.input_hists):
            inputs_od[idx] = get_hist_entry(*self._get_min_max(module.distiller_name, 'inputs', idx),
                                            module.input_hists[idx])

        output_od = get_hist_entry(*self._get_min_max(module.distiller_name, 'output'), module.output_hist)

        stats_od['inputs'] = inputs_od
        stats_od['output'] = output_od
        # Todo: 注意，每个收集器对象只负责指定指标的记录，因此，并不会冲突
        activation_stats[module.distiller_name] = stats_od

    def save(self, fname):
        hist_dict = self.value()

        if not fname.endswith('.pt'):
            fname = ".".join([fname, 'pt'])
        try:
            os.remove(fname)
        except OSError:
            pass

        torch.save(hist_dict, fname)

        if self.save_imgs:
            msglogger.info('Saving histogram images...')
            save_dir = os.path.join(os.path.split(fname)[0], 'histogram_imgs')
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            # noinspection PyTypeChecker
            def save_hist(layer_name, tensor_name, idx, bin_counts, bin_centroids, normed=True):
                if normed:
                    bin_counts = bin_counts / bin_counts.sum()
                plt.figure(figsize=(12, 12))
                plt.suptitle('\n'.join((layer_name, tensor_name)), fontsize=18, fontweight='bold')
                for subplt_idx, yscale in enumerate(['linear', 'log']):
                    plt.subplot(2, 1, subplt_idx + 1)
                    plt.fill_between(bin_centroids, bin_counts, step='mid', antialiased=False)
                    if yscale == 'linear':
                        plt.ylim(bottom=0)
                    plt.title(yscale + ' scale')
                    plt.yscale(yscale)
                    plt.xlabel('Activation Value')
                    plt.ylabel('Normalized Count')
                plt.tight_layout(rect=[0, 0, 1, 0.93])
                idx_str = '{:03d}'.format(idx)
                plt.savefig(os.path.join(save_dir, '-'.join((idx_str, layer_name, tensor_name)) + self.imgs_ext))
                plt.close()

            cnt = 0
            for layer_name, data in hist_dict.items():
                for idx, od in data['inputs'].items():
                    cnt += 1
                    save_hist(layer_name, 'input_{}'.format(idx), cnt, od['hist'], od['bin_centroids'], normed=True)
                od = data['output']
                cnt += 1
                save_hist(layer_name, 'output', cnt, od['hist'], od['bin_centroids'], normed=True)
            msglogger.info('Done')
        return fname


# 仅收集激活单元的值
class RawActivationsCollector(ActivationStatsCollector):
    def __init__(self, model, classes=None):
        super(RawActivationsCollector, self).__init__(model, "raw_acts", classes)

        _verify_no_dataparallel(model)

    def _activation_stats_cb(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            # 如果输出已经被量化，则进行解量化
            if output.is_quantized:
                module.raw_outputs.append(output.dequantize())
            else:
                # 执行收集数据的动作
                module.raw_outputs.append(output.cpu())

    def _start_counter(self, module):
        module.raw_outputs = []

    def _reset_counter(self, module):
        if hasattr(module, 'raw_outputs'):
            module.raw_outputs = []

    def _collect_activations_stats(self, module, activation_stats, name=''):
        if not hasattr(module, 'raw_outputs'):
            return
        # 如果raw_outputs是一个list，则将其堆叠起来
        if isinstance(module.raw_outputs, list) and len(module.raw_outputs) > 0:
            module.raw_outputs = torch.stack(module.raw_outputs)
        activation_stats[module.distiller_name] = module.raw_outputs

    def save(self, dir_name):
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for idx, (layer_name, raw_outputs) in enumerate(self.value().items()):
                idx_str = '{:03d}'.format(idx + 1)
                executor.submit(torch.save, raw_outputs, os.path.join(dir_name,
                                                                      '-'.join((idx_str, layer_name)) + '.pt'))

        return dir_name


# Todo: 以上都是定义的收集类，这里的是使用以上类的帮助函数

# 量化部分先跳过
def collect_quant_stats(model, test_fn, save_dir=None, classes=None, inplace_runtime_check=False,
                        disable_inplace_attrs=False, inplace_attr_names=('inplace',),
                        modules_to_collect=None):
    """
    为使用QuantCalibrationStatsCollector的类收集量化校验统计数据的帮助函数

    Args:
        model (nn.Module): 需要收集统计数据的模型
        test_fn (function): 模型的测试或评估函数，必须要有model参数，接收一个模型，其余的参数通过partial提前设定
        save_dir (str): 存储统计YAML文件的路径，如果为None，则存储到磁盘上
        classes (iterable): See QuantCalibrationStatsCollector
        inplace_runtime_check (bool): See QuantCalibrationStatsCollector
        disable_inplace_attrs (bool): See QuantCalibrationStatsCollector
        inplace_attr_names (iterable): See QuantCalibrationStatsCollector
        modules_to_collect (iterable): 需要收集统计数据的预定义的模块列表

    Returns:
        Dictionary with quantization stats (see QuantCalibrationStatsCollector for a description of the dictionary
        contents)
    """
    msglogger.info('Collecting quantization calibration stats for model')
    quant_stats_collector = QuantCalibrationStatsCollector(model, classes=classes,
                                                           inplace_runtime_check=inplace_runtime_check,
                                                           disable_inplace_attrs=disable_inplace_attrs,
                                                           inplace_attr_names=inplace_attr_names)
    with collector_context(quant_stats_collector, modules_to_collect):
        msglogger.info('Pass 1: Collecting min, max, avg_min, avg_max, mean')
        test_fn(model=model)
        # 收集拉普拉斯分布的统计数据
        msglogger.info('Pass 2: Collecting b, std parameters')
        quant_stats_collector.start_second_pass()
        test_fn(model=model)
        quant_stats_collector.stop_second_pass()

    msglogger.info('Stats collection complete')
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'acts_quantization_stats.yaml')
        quant_stats_collector.save(save_path)
        msglogger.info('Stats saved to ' + save_path)

    return quant_stats_collector.value()


def collect_histograms(model, test_fn, save_dir=None, activation_stats=None,
                       classes=None, nbins=2048, save_hist_imgs=False, hist_imgs_ext='.svg'):
    """
    Helper function for collecting activation histograms for a model using ActivationsHistogramCollector.
    Will perform 2 passes -
        Pass1: 收集需要的统计数据（如果提前传递激活单元的统计数据，则该步骤可以跳过）
        Pass2: 收集直方图
    Args:
        model (nn.Module): The model for which to collect histograms
        test_fn (function): Test/Evaluation function for the model. It must have an argument named 'model' that
          accepts the model. All other arguments should be set in advance (can be done using functools.partial), or
          they will be left with their default values.
        save_dir (str): Path to directory where histograms will be saved. If None then data will not be saved to disk.
        activation_stats (str / dict / None): Either a path to activation stats YAML file, or a dictionary containing
          the stats. The stats are expected to be in the same structure as generated by QuantCalibrationStatsCollector.
          If None, then a stats collection pass will be performed.
        classes: See ActivationsHistogramCollector
        nbins: See ActivationsHistogramCollector
        save_hist_imgs: See ActivationsHistogramCollector
        hist_imgs_ext: See ActivationsHistogramCollector

    Returns:
        Dictionary with histograms data (See ActivationsHistogramCollector for a description of the dictionary
        contents)
    """
    msglogger.info('Pass 1: Stats collection')
    if activation_stats is not None:
        msglogger.info('Pre-computed activation stats passed, skipping stats collection')
    else:
        # 收集量化的统计数据
        activation_stats = collect_quant_stats(model, test_fn, save_dir=save_dir, classes=classes,
                                               inplace_runtime_check=True, disable_inplace_attrs=True)

    msglogger.info('Pass 2: Histograms generation')
    # 创建一个直方图收集器对象
    histogram_collector = ActivationHistogramsCollector(model, activation_stats, classes=classes, nbins=nbins,
                                                        save_hist_imgs=save_hist_imgs, hist_imgs_ext=hist_imgs_ext)
    # Todo: with语句开始时调用一次collector_context(collector)方法，结束时再调用一次？
    with collector_context(histogram_collector):
        test_fn(model=model)
    msglogger.info('Histograms generation complete')
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'acts_histograms.pt')
        histogram_collector.save(save_path)
        msglogger.info("Histogram data saved to " + save_path)
        if save_hist_imgs:
            msglogger.info('Histogram images saved in ' + os.path.join(save_dir, 'histogram_imgs'))

    # 返回收集到的数据
    return histogram_collector.value()


# 收集纯输出
def collect_raw_outputs(model, test_fn, save_dir=None, classes=None):
    msglogger.info('Collecting raw layer outputs for model')
    collector = RawActivationsCollector(model, classes=classes)
    with collector_context(collector):
        test_fn(model=model)
    msglogger.info('Outputs collection complete')
    if save_dir is not None:
        msglogger.info('Saving outputs to disk...')
        save_path = os.path.join(save_dir, 'raw_outputs')
        collector.save(save_path)
        msglogger.info('Outputs saved to ' + save_path)
    return collector.value()


# Todo: 关于contextmanager的用法
@contextmanager
def collector_context(collector, modules_list=None):
    """激活单元收集器的上下文管理器"""
    if collector is not None:
        collector.reset().start(modules_list)
    yield collector
    if collector is not None:
        collector.stop()


@contextmanager
def collectors_context(collectors_dict):
    """收集器字典的上下文管理器"""
    if len(collectors_dict) == 0:
        yield collectors_dict
        return
    for collector in collectors_dict.values():
        # Todo: 不传入collector.classes参数，统计所有层的数据
        collector.reset().start()
    yield collectors_dict
    for collector in collectors_dict.values():
        collector.stop()


class TrainingProgressCollector(object):
    def __init__(self, stats=None):
        super(TrainingProgressCollector, self).__init__()
        if stats is None:
            stats = {}
        object.__setattr__(self, '_stats', stats)

    def __setattr__(self, name, value):
        stats = self.__dict__.get('_stats')
        stats[name] = value

    def __getattr__(self, name):
        if name in self.__dict__['_stats']:
            return self.__dict__['_stats'][name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def value(self):
        return self._stats
