from collections import namedtuple, OrderedDict
import re
import copy
import logging
import torch
import torch.nn as nn
import warnings
from typing import Callable, Optional
from copy import deepcopy
import federatedml.nn.backend.distiller.utils as utils
from federatedml.nn.backend.distiller.summary_graph import SummaryGraph

QBits = namedtuple('QBits', ['acts', 'wts', 'bias'])

FP_BKP_PREFIX = 'float_'
msglogger = logging.getLogger('quant_logger')


def has_bias(module):
    return hasattr(module, 'bias') and module.bias is not None


# 对浮点类型的参数进行备份
def hack_float_backup_parameter(module, name, num_bits):
    try:
        param = dict(module.named_parameters())[name]
        # 获取参数的id
        param_id = id(param)
    except KeyError:
        raise ValueError('Module has no Parameter named' + name)
    # Todo: 对原来的parameter属性进行备份
    module.register_parameter(FP_BKP_PREFIX + name, param)
    # 确保参数的id不发生变化
    assert id(getattr(module, FP_BKP_PREFIX + name)) == param_id
    delattr(module, name)
    # 注册量化参数
    module.register_buffer(name, torch.zeros_like(param))
    # 是否已经进行初始化的标识
    first = False
    # Todo: 注册repr_mod属性，but，repr_mod属性的含义是什么？表征模式？
    # Todo: 进行debug，观察相关的属性值
    if not hasattr(module, 'repr_mod'):
        setattr(module, 'repr_mod', ', \nDistiller_QuantAwareTrain: ')
        first = True
        module.original_extra_repr = module.extra_repr
        module.extra_repr = lambda: module.original_extra_repr() + module.repr_mod
    # 如果不是初始化
    if not first:
        module.repr_mod += ' ; '
    module.repr_mod += '{0} --> {1} bits'.format(name, num_bits)


class _ParamToQuant(object):
    def __init__(self, module, module_name, fp_attr_name, q_attr_name, num_bits):
        self.module = module
        self.module_name = module_name
        self.fp_attr_name = fp_attr_name
        self.q_attr_name = q_attr_name
        self.num_bits = num_bits

    def __repr__(self):
        return "ParamToQuant(module_name=%s,num_bits=%s)" % (self.module_name, self.num_bits)


class Quantizer(object):
    """
    量化器的基类
    参数：
        model(torch.nn.Module): 需要量化的模型
        # Todo: 为什么不能直接对model进行修改？
        optimizer(torch.optim.Optimizer): 优化器的实例，当量化器需要对以存在的模型进行修改或添加新的参数时，需要该实例
                                        当train_with_fp_copy为真时，优化器实例不能为None
        bits_activations/weights/bias (int): 量化每个张量时使用的默认bit数，值为None表示不使用量化
        # Todo: 该字段结合具体的实例进行分析
        overrides (OrderedDict): 将模型各个层的常规表示映射到具有默认值的重写的字典。
                                该字典中的key是量化器接受的参数名称，以便于初始化
                                默认支持bits_activations、bits_bias、bits_weights
                                有序字典的数据类型使得可以处理重复的名称模式
        # 训练过程再梳理
        train_with_fp_copy (bool): 当设为true时，将修改有权重的层，同时保持量化和浮点两个版本，基点是在fp_weights
                                    1. q_weights = quantize(fp_weights)
                                    2. 使用q_weights执行前向传播
                                    3. 反向传播阶段：
                                        3.1 计算关于q_weights的梯度
                                        3.2 对反响传播执行步1的量化操作
                                    4. 使用步3.2中计算的梯度更新fp_weights
        注意：overrides字典假设键不是nn.DataParallel中的模型名称，即没有module.的前缀
        例如：module.conv1 -> OrderedDict([('conv1', OrderedDict(...))])
    """

    def __init__(self, model, optimizer=None,
                 bits_activations=None, bits_weights=None, bits_bias=None,
                 overrides=None, train_with_fp_copy=False):
        if overrides is None:
            overrides = OrderedDict()
        if not isinstance(overrides, OrderedDict):
            raise TypeError('overrides must be an instance of collections.OrderedDict or None')

        if train_with_fp_copy and optimizer is None:
            raise ValueError('optimizer cannot be None when train_with_fp_copy is True')
        # Todo: 该map的含义是什么？
        # 当调用prepare_model()方法时，将向其中注入数据
        self.adjacency_map = None
        # 使用命名元组来存放量化的比特位数
        self.default_qbits = QBits(acts=bits_activations, wts=bits_weights, bias=bits_bias)
        self.overrides = overrides

        self.model = model
        self.optimizer = optimizer

        # 将模型中的量化数据存储起来，以便在恢复模型上重新应用量化器
        self.model.quantizer_metadata = {
            'type': type(self),
            'params': {
                'bits_activations': bits_activations,
                'bits_weights': bits_weights,
                'bits_bias': bits_bias,
                'overrides': copy.deepcopy(overrides)
            }
        }
        # k：模型中的每个模块;v：该模块的量化信息配置
        for k, v in self.overrides.items():
            if any(old_bits_key in v.keys() for old_bits_key in ['acts', 'wts', 'bias']):
                raise ValueError("Using 'acts' / 'wts' / 'bias' to specify bit-width overrides is deprecated.\n"
                                 "Please use the full parameter names: "
                                 "'bits_activations' / 'bits_weights' / 'bits_bias'")
            qbits = QBits(acts=v.pop('bits_activations', self.default_qbits.acts),
                          wts=v.pop('bits_weights', self.default_qbits.wts),
                          bias=v.pop('bits_bias', self.default_qbits.bias))
            # 对该模块的bits属性赋值
            v['bits'] = qbits
        # 基于默认和重写的配置准备从每个层到QBits的显式的映射
        patterns = []
        regex_overrides = None
        if overrides:
            patterns = list(overrides.keys())
            # 使用竖线将列表连接成字符串
            regex_overrides_str = '|'.join(['(^{0}$)'.format(pattern) for pattern in patterns])
            regex_overrides = re.compile(regex_overrides_str)
        # type: OrderedDict[str,QBits]
        self.module_qbits_map = {}
        self.module_overrides_map = {}
        for module_full_name, module in model.named_modules():
            # 考虑数据并行的模型场景，模型名称前面有module.
            name_to_match = module_full_name.replace('module.', '', 1)
            qbits = self.default_qbits
            # Todo: 关于正则项匹配的细节？结合量化实例去理解
            # 获取重写的量化配置
            overrides_entry = self.overrides.get(name_to_match, OrderedDict())
            if regex_overrides:
                m_overrides = regex_overrides.match(name_to_match)
                if m_overrides:
                    group_idx = 0
                    groups = m_overrides.groups()
                    while groups[group_idx] is None:
                        group_idx += 1
                    # 得到对应的overrides_entry对应项，或者根据匹配模式进行的配置
                    overrides_entry = copy.deepcopy(overrides_entry or self.overrides[patterns[group_idx]])
                    qbits = overrides_entry.pop('bits', self.default_qbits)
            self._add_qbits_entry(module_full_name, type(module), qbits)
            self._add_override_entry(module_full_name, overrides_entry)
        # Todo: 替换工厂和替换函数在量化策略中的作用？
        # 从模块类型到函数的映射产生一个适合量化的可以替代的模型，它由子类进行数据填充
        # 未指明层的类型会产生一个None值
        self.replacement_factory = OrderedDict([(nn.Identity, None)])
        self.default_replacement_fn = None
        self.replacement_blacklist = []
        # 参数量化函数的指针，在训练过程中调用，在子类中填充数据
        self.param_quantization_fn = None
        self.train_with_fp_copy = train_with_fp_copy
        self.params_to_quantize = []
        # 包含可替换模块及其名称的字典
        self.modules_processed = OrderedDict()
        self.modules_processed_args = OrderedDict()

        self.prepared = False

    def _add_qbits_entry(self, module_name, module_type, qbits):
        # 如果不支持当前模块的wts量化，则指定以acts的量化
        if module_type not in [nn.Conv2d, nn.Conv3d, nn.Linear, nn.Embedding]:
            qbits = QBits(acts=qbits.acts, wts=None, bias=None)
        self.module_qbits_map[module_name] = qbits

    def _add_override_entry(self, module_name, entry):
        self.module_overrides_map[module_name] = entry

    def prepare_model(self, dummy_input=None):
        """遍历模块类型，根据bit-width和__init__()函数提供的重写配置以及量化器子类定义的替换工厂替换需要量化的子模块

        注意，如果模型中多个子模块的含义是同一个模块时，只替换一次
        例如：
            shared_relu = nn.ReLU
            self.relu1 = shared_relu
            self.relu2 = shared_relu
        当遍历到该模块时，遇到self.relu1时将生成一个替换，称其为new_relu1；当遇到self.relu2时，使用new_relu1的索引进行替换
        任何对self.relu2重写的配置将忽略，并且抛出一个警告信息
        :param dummy_input:
        :return:
        """
        if self.prepared:
            raise RuntimeError('prepare_model仅可调用一次')
        # 获取计算图
        self.model.quantizer_metadata["dummy_input"] = dummy_input
        if dummy_input is not None:
            summary_graph = SummaryGraph(self.model, dummy_input)
            self.adjacency_map = summary_graph.adjacency_map(dedicated_modules_only=False)
            del summary_graph
        model_device = utils.model_device(self.model)
        # 根据dummy_input预先准备模型
        # 具体逻辑由其子类实现
        self._pre_prepare_model(dummy_input)
        self._pre_process_container(self.model)

        # 遍历模型中的命名模块
        for module_name, module in self.model.named_modules():
            # 获取量化模块的qbits
            qbits = self.module_qbits_map[module_name]
            # 获取当前模块中的具体参数
            cur_parameters = dict(module.named_parameters())
            for param_name, param in cur_parameters.items():
                n_bits = qbits.bias if param_name.endswith('bias') else qbits.wts
                if n_bits is None:
                    continue
                fp_attr_name = param_name
                if self.train_with_fp_copy:
                    # 备份浮点型的参数
                    hack_float_backup_parameter(module, param_name, n_bits)
                    fp_attr_name = FP_BKP_PREFIX + param_name
                # 添加需要量化的参数
                self.params_to_quantize.append(_ParamToQuant(module, module_name, fp_attr_name, param_name, n_bits))
                param_full_name = '.'.join([module_name, param_name])
                msglogger.debug(
                    "Parameter '{0}' will be quantized to {1} bits".format(param_full_name, n_bits))
        if self.optimizer:
            for pg in self._get_new_optimizer_params_groups():
                self.optimizer.add_param_group(pg)
        self._post_prepare_model()
        # Todo: 为什么需要重新转移？
        # 将模型重新转移到它所在的设备，防止量化器创建新的参数/缓冲
        self.model.to(model_device)
        utils.assign_layer_fq_names(self.model)
        self.prepared = True

        msglogger.debug('Quantized model:\n\n{0}\n'.format(self.model))

    def _pre_prepare_model(self, dummy_input):
        pass

    def _pre_process_container(self, container, prefix=''):
        """预处理容器

        :param container: 容器
        :param prefix: 前缀，比如浮点前缀FP_PREFIX
        :return:
        """

        # 记录替换的消息
        def replace_msg(module_name, modules=None):
            # 这里modules[0]是被替换的模块，而modules[1]是替换模块
            msglogger.debug('Module ' + module_name)
            if modules:
                msglogger.debug('\tReplacing: {}.{}'.format(modules[0].__module__, modules[0].__class__.__name__))
                msglogger.debug('\tWith:      {}.{}'.format(modules[1].__module__, modules[1].__class__.__name__))
            else:
                msglogger.debug('\tSkipping')

        # 对模型迭代，并加入量化函数
        # Todo: 对named_children()迭代而不是named_modules()进行迭代，两个方法的区别是什么？
        for name, module in container.named_children():
            full_name = prefix + name
            # 替换黑名单，即不需要替换的模块
            if isinstance(module, tuple(self.replacement_blacklist)):
                replace_msg(full_name)
                continue
            # 如果模块之前处理过
            if module in self.modules_processed:
                previous_name, previous_wrapper = self.modules_processed[module]
                # 类注释中提到的多次遇到的模块，使用之前生成替换对象的引用进行替换
                warnings.warn("Module '{0}' references to same module as '{1}'."
                              ' Replacing with reference the same wrapper.'.format(full_name, previous_name),
                              UserWarning)
                if previous_wrapper:
                    replace_msg(full_name, (module, previous_wrapper))
                    # 设置引用即可
                    setattr(container, name, previous_wrapper)
                    # Todo: 之前替换过，但引用为None，是哪种情况，见下方的处理情况
                else:
                    replace_msg(full_name)
                continue
            # 获取模块对应的量化信息qbits
            current_qbits = self.module_qbits_map[full_name]
            if current_qbits.acts is None and current_qbits.wts is None and not self.module_overrides_map[full_name]:
                replace_msg(full_name)
                # 没有找到该模块对应的配置信息，因此，不进行替换
                self.modules_processed[module] = full_name, None
            else:
                replace_fn = self.replacement_factory.get(type(module),
                                                          self.default_replacement_fn)  # type: Optional[Callable]
                # 如果没有指定过替换函数，则不进行替换
                if replace_fn is not None:
                    # 替换函数的参数在self.module_overrides_map中
                    valid_kwargs, invalid_kwargs = utils.filter_kwargs(self.module_overrides_map[full_name],
                                                                       replace_fn)
                    if invalid_kwargs:
                        raise TypeError("""Quantizer of type %s doesn't accept \"%s\" 
                                            as override arguments for %s. Allowed kwargs: %s"""
                                        % (type(self), list(invalid_kwargs), type(module), list(valid_kwargs)))
                    # 执行替换操作
                    new_module = replace_fn(module, full_name, self.module_qbits_map, **valid_kwargs)
                    if new_module != module:
                        replace_msg(full_name, (module, new_module))
                        self.modules_processed[module] = full_name, new_module
                        # 保存量化的比特数以及参数
                        valid_args = full_name, deepcopy(self.module_qbits_map)
                        self.modules_processed_args[full_name] = valid_args, valid_kwargs
                        # 给容器设置替换后的模型
                        setattr(container, name, new_module)
                        # Todo: 为什么会出现这种情况？使用一个容器来替换一个叶子模型？
                        if not utils.has_children(module) and utils.has_children(new_module):
                            for sub_module_name, sub_module in new_module.named_modules():
                                # Todo: 注意：_add_qbits_entry实际上就是向module_qbits_map中添加
                                self._add_qbits_entry(full_name + '.' + sub_module_name, type(sub_module),
                                                      current_qbits)
                            self.module_qbits_map[full_name] = QBits(acts=current_qbits.acts, wts=None, bias=None)
                    # 没有替换成功
                    else:
                        replace_msg(full_name)
                        self.modules_processed[module] = full_name, None
            # 递归处理子模块
            if utils.has_children(module):
                self._pre_process_container(module, full_name + '.')

    def _get_new_optimizer_params_groups(self):

        return list()

    def _post_prepare_model(self):
        pass

    def quantize_params(self):
        for ptq in self.params_to_quantize:
            q_param = self.param_quantization_fn(getattr(ptq.module, ptq.fp_attr_name), ptq)
            if self.train_with_fp_copy:
                # 给模块设置属性名称
                setattr(ptq.module, ptq.q_attr_name, q_param)
            else:
                getattr(ptq.module, ptq.q_attr_name).data = q_param.data
