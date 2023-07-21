import typing
import warnings
from numbers import Number
from collections import namedtuple

import torch
import torch.nn as nn
import federatedml.nn.backend.multi_label.prunners.depgraph.ops as ops

# Todo: 可以先封装一下，然后进行剪枝后，重新配置参数

import federatedml.nn.backend.multi_label.prunners.depgraph.function as function


# 依赖图中节点的定义
class Node(object):
    def __init__(self, module: nn.Module, grad_fn, name: str = None):
        self.inputs = []
        self.outputs = []
        self.module = module
        self.grad_fn = grad_fn
        self._name = name
        self.type = ops.module2type(module)
        self.class_type = module.__class__

        # 依赖图相关
        self.dependencies = []
        # Todo: 启用索引映射是什么意思？
        self.enable_index_mapping = True
        self.pruning_dim = -1

    @property
    def name(self):
        if self._name is None:
            return str(self.module)
        else:
            fmt = self._name
            if self.type != ops.OPTYPE.PARAMETER:
                fmt += " ({})".format(str(self.module))
            return fmt

    # 为节点添加输入节点
    def add_input(self, node, allow_duplicated=False):
        # 如果允许重复，则直接添加；否则，判重后添加
        if allow_duplicated:
            self.inputs.append(node)
        else:
            if node not in self.inputs:
                self.inputs.append(node)

    # 为节点添加输出节点
    def add_output(self, node, allow_duplicated=False):
        if allow_duplicated is True:
            self.outputs.append(node)
        else:
            if node not in self.outputs:
                self.outputs.append(node)

    def __repr__(self):
        return "<Node: ({})>".format(self.name)

    def __str__(self):
        return "<Node: ({})>".format(self.name)

    def details(self):
        fmt = "-" * 32 + "\n"
        fmt += "<Node: ({})>\n".format(self.name)
        fmt += " " * 4 + "IN:\n"
        for in_node in self.inputs:
            fmt += " " * 8 + "{}\n".format(in_node)
        fmt += " " * 4 + "OUT:\n"
        for out_node in self.outputs:
            fmt += " " * 8 + "{}\n".format(out_node)
        fmt += " " * 4 + "DEP:\n"
        for dep in self.dependencies:
            fmt += " " * 8 + "{}\n".format(dep)
        fmt += "\tEnable_index_mapping={}\n".format(
            self.enable_index_mapping)
        fmt = "-" * 32 + "\n"
        return fmt


# 为了提高可读性，创建该dummy class
class Edge():
    pass


class Dependency(Edge):
    # 关于下面解释：对source的trigger剪枝触发了对target的handler剪枝
    def __init__(self,
                 trigger: typing.Callable,
                 handler: typing.Callable,
                 source: Node,
                 target: None):
        self.trigger = trigger
        self.handler = handler
        self.source = source
        self.target = target
        self.index_mapping = [None, None]

    # idxs表示剪枝索引，如target模块对应下标为idxs的通道
    def __call__(self, masks, idxs: list):
        # 剪枝维度，是什么意思？
        self.handler.__self__.pruning_dim = self.target.pruning_dim
        self.handler(
            self.target.module,
            masks,
            idxs
        )

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "{} on {} => {} on {}".format(
            "None" if self.trigger is None else self.trigger.__name__,
            self.source.name,
            self.handler.__name__,
            self.target.name,
        )

    # 验证该依赖是否由pruning_fn所触发
    def is_triggered_by(self, pruning_fn):
        return pruning_fn == self.trigger

    def __eq__(self, other):
        return (self.source == other.source
                and self.trigger == other.trigger
                and self.handler == other.handler
                and self.target == other.target)

    def __hash__(self):
        return hash((self.source, self.target, self.trigger, self.handler))


# 创建命名元组
GroupItem = namedtuple('GroupItem', ['dep', 'idxs'])


# 表示参数组的类
class Group(object):
    def __init__(self):
        self._group = list()
        self._DG = None

    # 对组内所有耦合的层进行剪枝，剪枝的通道索引是idxs
    # Todo: 理清该剪枝方法的脉络
    def prune(self, name2masks, idxs=None, record_history=False):
        if idxs is not None:
            # 需要剪枝的模块，这里从第一个依赖的目标模块出发
            module = self._group[0].dep.target.module
            # 剪枝函数，就是handler
            pruning_fn = self._group[0].dep.handler
            # 给定入口模块、剪枝函数和剪枝的通道索引
            new_group = self._DG.get_pruning_group(module, pruning_fn, idxs)
            # 执行剪枝
            new_group.prune()
        else:
            # 未指定需要剪枝的通道索引
            # Todo: 对组内每一个依赖，执行dep(idxs)
            for dep, idxs in self._group:
                # 如果是对parameter进行剪枝
                if dep.target.type == ops.OPTYPE.PARAMETER:
                    old_parameter = dep.target.module
                    name = self._DG._param_to_name[old_parameter]
                    self._DG._param_to_name.pop(old_parameter)
                    pruned_parameter = dep(idxs)
                    path = name.split('.')
                    module = self._DG.model
                    for p in path[:-1]:
                        module = getattr(module, p)
                    setattr(module, path[-1], pruned_parameter)
                    self._DG._param_to_name[pruned_parameter] = name
                    self._DG.module2node[pruned_parameter] = self._DG.module2node.pop(old_parameter)
                    self._DG.module2node[pruned_parameter].module = pruned_parameter
                else:
                    # 正常的剪枝过程

                    # 获取模块名称
                    module_name = dep.target.name.split(' ')[0]
                    masks = []
                    weight_name = f'{module_name}.weight'
                    if weight_name in name2masks:
                        masks.append(name2masks[weight_name])
                    bias_name = f'{module_name}.bias'
                    if bias_name in name2masks:
                        masks.append(name2masks[bias_name])
                    dep(masks, idxs)
                    # print('pause')
        if record_history:
            root_module, pruning_fn, root_pruning_idx = self[0][0].target.module, self[0][0].trigger, self[0][1]
            root_module_name = self._DG._module2name[root_module]
            self._DG._pruning_history.append(
                [root_module_name, self._DG.is_out_channel_pruning_fn(pruning_fn), root_pruning_idx])

    # 添加依赖
    def add_dep(self, dep, idxs):
        self._group.append(GroupItem(dep=dep, idxs=idxs))

    def __getitem__(self, k):
        return self._group[k]

    @property
    def items(self):
        return self._group

    # 验证组内是否存在dep
    def has_dep(self, dep):
        for _dep, _ in self._group:
            if dep == _dep:
                return True
        return False

    # 验证组内是否存在与dep层等价的剪枝操作
    def has_pruning_op(self, dep, idxs):
        for _dep, _idxs in self._group:
            if (_dep.target == dep.target
                    and _dep.handler == dep.handler
                    and _idxs == idxs):
                return True
        return False

    # 添加并合并
    def add_and_merge(self, dep, idxs):
        for i, (_dep, _idxs) in enumerate(self._group):
            # 如果需要对target进行handler剪枝，则将剪枝通道进行合并
            if _dep.target == dep.target and _dep.handler == dep.handler:
                self._group[i] = (_dep, list(set(_idxs + idxs)))
                return
        # 是新的剪枝行为，直接添加
        self.add_dep(dep, idxs)

    def __str__(self):
        fmt = ""
        fmt += "\n" + "-" * 32 + "\n"
        fmt += " " * 10 + "Pruning Group"
        fmt += "\n" + "-" * 32 + "\n"
        for i, (dep, idxs) in enumerate(self._group):
            fmt += "[{}] {}, #idxs={}\n".format(i, dep, len(idxs))
        fmt += "-" * 32 + "\n"
        return fmt

    def details(self):
        fmt = ""
        fmt += "\n" + "-" * 32 + "\n"
        fmt += " " * 10 + "Pruning Group"
        fmt += "\n" + "-" * 32 + "\n"
        for i, (dep, idxs) in enumerate(self._group):
            if i == 0:
                fmt += "[{}] {}, idxs={} (Pruning Root)\n".format(i, dep, idxs)
            else:
                fmt += "[{}] {}, idxs={}\n".format(i, dep, idxs)
        fmt += "-" * 32 + "\n"
        return fmt

    # 旧的接口，使用group.prune()代替
    def exec(self):
        self.prune()

    def __call__(self):
        return self.prune()


# 依赖图的表示类
class DependencyGraph(object):
    def __init__(self):
        # 默认的剪枝器
        _dummy_pruners = {
            ops.OPTYPE.CONCAT: ops.ConcatPruner(),
            ops.OPTYPE.SPLIT: ops.SplitPruner(),
            ops.OPTYPE.ELEMENTWISE: ops.ElementWisePruner(),
            ops.OPTYPE.RESHAPE: ops.ReshapePruner(),
            ops.OPTYPE.CUSTOMIZED: None,
        }
        self.REGISTERED_PRUNERS = function.PrunerBox.copy()
        self.REGISTERED_PRUNERS.update(_dummy_pruners)
        self.CUSTOMIZED_PRUNERS = {}
        # 需要忽略的层
        self.IGNORED_LAYERS = []

        # Todo: 使用集合，方便判定剪枝函数fn属于对输入通道的剪枝函数还是对输出通道的剪枝函数
        # 对输入通道进行剪枝的函数
        self._in_channel_pruning_fn = set(
            [p.prune_in_channels for p in self.REGISTERED_PRUNERS.values() if p is not None] +
            [p.prune_in_channels for p in self.CUSTOMIZED_PRUNERS.values() if p is not None]
        )

        # 对输出通道进行剪枝的函数
        self._out_channel_pruning_fn = set(
            [p.prune_out_channels for p in self.REGISTERED_PRUNERS.values() if p is not None] +
            [p.prune_out_channels for p in self.CUSTOMIZED_PRUNERS.values() if p is not None]
        )

        self._op_id = 0

        # 维护剪枝历史
        self._pruning_history = []

    def pruning_history(self):
        return self._pruning_history

    # Todo: 加载剪枝历史
    def load_pruning_history(self, pruning_history):
        self._pruning_history = pruning_history
        for module_name, is_out_channel_pruning, pruning_idx in self._pruning_history:
            module = self.model
            # 这个for循环的作用是获取最后一个模块吗？
            for n in module_name.split('.'):
                module = getattr(module, n)
            pruner = self.get_pruner_of_module(module)
            if is_out_channel_pruning:
                pruning_fn = pruner.prune_out_channels
            else:
                pruning_fn = pruner.prune_in_channels
            group = self.get_pruning_group(module, pruning_fn, pruning_idx)
            group.prune(record_history=False)

    # 通过tracing构建依赖图
    def build_dependency(self,
                         model: torch.nn.Module,
                         example_inputs: typing.Union[torch.Tensor, typing.Sequence],
                         forward_fn: typing.Callable[[
                             torch.nn.Module, typing.Union[torch.Tensor, typing.Sequence]], torch.Tensor] = None,
                         output_transform: typing.Callable = None,
                         unwrapped_parameters: typing.Dict[nn.Parameter, int] = None,
                         customized_pruners: typing.Dict[typing.Any,
                         function.BasePruningFunc] = None,
                         verbose: bool = True,
                         ):
        # 详细输出
        self.verbose = verbose
        # 参考的网络模型
        self.model = model
        # 模块到名称的映射
        self._module2name = {module: name for (name, module) in model.named_modules()}
        if customized_pruners is not None:
            for customized_module, customized_pruner in customized_pruners.items():
                self.register_customized_layer(customized_module, customized_pruner)
        # 遍历自定义剪枝器中的层类型
        # Todo: 忽略自定义层的所有子模块 --> 为什么忽略呢？
        for layer_type in self.CUSTOMIZED_PRUNERS.keys():
            for m in self.model.modules():
                # 如果m是对应类型的层，对于m的模块，将其添加到忽略层中
                if isinstance(m, layer_type):
                    for sub_module in m.modules():
                        if sub_module != m:
                            self.IGNORED_LAYERS.append(sub_module)
        # 检测未包装的参数
        wrapped_parameters = []
        prunable_module_types = self.REGISTERED_PRUNERS.keys()
        for m in self.model.modules():
            op_type = ops.module2type(m)
            # m是可剪枝模块，且非ELEMENT_WISE类型或者m属于自定义剪枝层
            if op_type in prunable_module_types and op_type != ops.OPTYPE.ELEMENTWISE or m.__class__ in self.CUSTOMIZED_PRUNERS.keys():
                wrapped_parameters.extend(list(m.parameters()))
        unwrapped_detected = []
        # 存储未包装参数到名称的映射
        _param_to_name = {}
        for name, p in self.model.named_parameters():
            is_wrapped = False
            for p_wrapped in wrapped_parameters:
                if p is p_wrapped:
                    is_wrapped = True
                    break
            if not is_wrapped:
                unwrapped_detected.append(p)
                _param_to_name[p] = name
        if unwrapped_parameters is None:
            unwrapped_parameters = []
        self._param_to_name = _param_to_name
        unwrapped_detected = list(set(unwrapped_detected) - set([p for (p, _) in unwrapped_parameters]))
        if len(unwrapped_detected) > 0 and self.verbose:
            warnings.warn(
                "Unwrapped parameters detected: {}.\n Torch-Pruning will prune the last non-singleton dimension of a parameter. If you wish to customize this behavior, please provide an unwrapped_parameters argument.".format(
                    [_param_to_name[p] for p in unwrapped_detected]))
        # 对于检测到的未包装的参数
        for p in unwrapped_detected:
            def last_non_singleton_dim(tensor):
                non_singleton_dims = [i for i, s in enumerate(tensor.shape) if s > 1]
                return non_singleton_dims[-1] if non_singleton_dims else None

            # 获取最后一个大于1的维度
            pruning_dim = last_non_singleton_dim(p)
            if pruning_dim is not None:
                unwrapped_parameters.append(UnwrappedParameters(parameters=p, pruning_dim=pruning_dim))
        self.unwrapped_parameters = unwrapped_parameters
        self.module2node = self._trace(
            model, example_inputs, forward_fn, output_transform=output_transform
        )

        self._build_dependency(self.module2node)
        # 对SPLIT层的相关处理，这里先注释掉
        # self._init_shape_information()
        # self.update_index_mapping()
        return self

    def _init_shape_information(self):
        for module, node in self.module2node.items():
            if node.type == ops.OPTYPE.SPLIT:
                pass

    def update_index_mapping(self):
        for module, node in self.module2node.items():
            if node.type == ops.OPTYPE.RESHAPE:
                self._update_reshape_index_mapping(node)

    def _update_reshape_index_mapping(self, reshape_node: Node):
        pass

    # 注册自定义的剪枝器
    def register_customized_layer(self, layer_type: typing.Type, layer_pruner: function.BasePruningFunc):
        self.CUSTOMIZED_PRUNERS[layer_type] = layer_pruner
        # 更新缓存
        self._in_channel_pruning_fn = set(
            [p.prune_in_channels for p in self.REGISTERED_PRUNERS.values() if p is not None] + [p.prune_in_channels for
                                                                                                p in
                                                                                                self.CUSTOMIZED_PRUNERS.values()
                                                                                                if p is not None])
        self._out_channel_pruning_fn = set(
            [p.prune_out_channels for p in self.REGISTERED_PRUNERS.values() if p is not None] + [p.prune_out_channels
                                                                                                 for p in
                                                                                                 self.CUSTOMIZED_PRUNERS.values()
                                                                                                 if p is not None])

    def _trace(self, model, example_inputs, forward_fn, output_transform):
        model.eval()
        # 建立从梯度函数到模块的映射
        gradfn2module = {}
        # 存储已经访问过的模块，以及访问次数
        visited = {}
        self._2d_4d = True

        # 钩子函数的参数包括：模块、模块的输入、模块的输出
        def _record_grad_fn(module, inputs, outputs):
            # 记录访问次数
            if module not in visited:
                visited[module] = 1
            else:
                visited[module] += 1
            # 线性层且输出形状是3维的
            if isinstance(module, nn.Linear) and len(outputs.shape) == 3:
                self._2d_4d = False

            if isinstance(outputs, tuple):
                outputs = outputs[0]
            # LSTM相关，先跳过
            if isinstance(outputs, torch.nn.utils.rnn.PackedSequence):
                outputs = outputs.data
            # 维护梯度函数到模块的信息
            gradfn2module[outputs.grad_fn] = module

        registered_types = tuple(ops.type2class(t) for t in self.REGISTERED_PRUNERS.keys()) + tuple(
            self.CUSTOMIZED_PRUNERS.keys())
        # 注册前向传播时的钩子函数
        hooks = [
            m.register_forward_hook(_record_grad_fn)
            for m in model.modules()
            if (isinstance(m, registered_types) and m not in self.IGNORED_LAYERS)
        ]
        if forward_fn is not None:
            out = forward_fn(model, example_inputs)
        elif isinstance(example_inputs, dict):
            out = model(**example_inputs)
        else:
            try:
                out = model(example_inputs)
            except:
                out = model(*example_inputs)
        # 移除模型中已经注册钩子函数
        for hook in hooks:
            hook.remove()
        # 对于递归的模块或层
        reused = [m for (m, count) in visited.items() if count > 1]

        # 构建图
        if output_transform is not None:
            out = output_transform(out)
        module2node = {}
        for o in self.flatten_as_list(out):
            self._trace_computational_graph(
                module2node, o.grad_fn, gradfn2module, reused
            )
        # 对ViT剪枝的部分，跳过
        return module2node

    # 追踪计算图
    def _trace_computational_graph(self, module2node, grad_fn_root, gradfn2module, reused):
        def create_node_if_not_exists(grad_fn):
            module = gradfn2module.get(grad_fn, None)
            if module is not None and module in module2node and module not in reused:
                return module2node[module]
            # 1. 链接梯度函数和模块
            if module is None:
                if not hasattr(grad_fn, "name"):
                    module = ops._ElementWiseOp(self._op_id, "Unknown")
                    self._op_id += 1
                elif "catbackward" in grad_fn.name().lower():
                    module = ops._ConcatOp(self._op_id)
                    self._op_id += 1
                # 划分split操作
                elif "split" in grad_fn.name().lower():
                    module = ops._SplitOp(self._op_id)
                    self._op_id += 1
                # view或reshape操作
                elif "view" in grad_fn.name().lower() or 'reshape' in grad_fn.name().lower():
                    module = ops._ReshapeOp(self._op_id)
                    self._op_id += 1
                else:
                    module = ops._ElementWiseOp(self._op_id, grad_fn.name())
                    self._op_id += 1
                gradfn2module[grad_fn] = module
            # 2. 链接模块和节点
            if module not in module2node:
                # 创建新节点
                node = Node(
                    module=module,
                    grad_fn=grad_fn,
                    name=self._module2name.get(module, None)
                )
                # 如果模块属于自定义剪枝器的剪枝层
                if type(module) in self.CUSTOMIZED_PRUNERS:
                    node.type = ops.OPTYPE.CUSTOMIZED
                module2node[module] = node
            else:
                node = module2node[module]
            return node

        # 处理栈
        processing_stack = [grad_fn_root]
        visited = set()
        visited_as_output_node = set()
        # 当栈非空
        while len(processing_stack) > 0:
            grad_fn = processing_stack.pop(-1)
            if grad_fn in visited:
                continue
            # 为该梯度函数创建节点
            node = create_node_if_not_exists(grad_fn=grad_fn)
            # 如果存在未处理的调用
            if hasattr(grad_fn, 'next_functions'):
                for f in grad_fn.next_functions:
                    if f[0] is not None:
                        # 对累加梯度的特殊处理
                        if (
                                hasattr(f[0], "name")
                                and "accumulategrad" in f[0].name().lower()
                        ):  # 一个叶子变量
                            is_unwrapped_param = False
                            for (j, (p, dim)) in enumerate(self.unwrapped_parameters):
                                if f[0].variable is p:
                                    is_unwrapped_param = True
                                    gradfn2module[f[0]] = p
                                    self._module2name[p] = "UnwrappedParameter_{} ({})".format(j, p.shape)
                            if not is_unwrapped_param:
                                continue
                        # 由于f[0]是反向传播过程中的next_function，因此，对应的节点是当前节点的输入节点
                        input_node = create_node_if_not_exists(f[0])
                        node.add_input(input_node, allow_duplicated=False)
                        input_node.add_output(node, allow_duplicated=False)
                        # 加入到处理栈中
                        processing_stack.append(f[0])
            visited.add(grad_fn)
            visited_as_output_node.add(node)
        for (param, dim) in self.unwrapped_parameters:
            module2node[param].pruing_dim = dim
        return module2node

    def _build_dependency(self, module2node):
        for _, node in module2node.items():
            # 规则1：层间依赖
            for in_node in node.inputs:
                handler = self.get_pruner_of_module(in_node.module).prune_out_channels
                trigger = self.get_pruner_of_module(node.module).prune_in_channels
                # 对当前节点的输入通道进行剪枝会触发对输入节点的输出通道剪枝
                dep = Dependency(
                    trigger=trigger, handler=handler, source=node, target=in_node
                )
                node.dependencies.append(dep)
            for out_node in node.outputs:
                trigger = self.get_pruner_of_module(node.module).prune_out_channels
                handler = self.get_pruner_of_module(out_node.module).prune_in_channels
                # 对当前节点的输出通道进行剪枝会触发对输出节点的输入通道剪枝
                dep = Dependency(
                    trigger=trigger, handler=handler, source=node, target=out_node
                )
                node.dependencies.append(dep)

    # 根据模块的类型获取对应的剪枝器
    def get_pruner_of_module(self, module):
        p = self.CUSTOMIZED_PRUNERS.get(module.__class__, None)
        if p is None:
            p = self.REGISTERED_PRUNERS.get(ops.module2type(module), None)
        return p

    def flatten_as_list(self, obj):
        if isinstance(obj, torch.Tensor):
            return [obj]
        elif isinstance(obj, (list, tuple)):
            flattened_list = []
            for sub_obj in obj:
                flattened_list.extend(self.flatten_as_list(sub_obj))
            return flattened_list
        elif isinstance(obj, dict):
            flattened_list = []
            for sub_obj in obj.values():
                flattened_list.extend(self.flatten_as_list(sub_obj))
            return flattened_list
        else:
            return obj

    # 获取剪枝函数对应的剪枝组
    def get_pruning_group(
            self,
            module: nn.Module,
            pruning_fn: typing.Callable,
            idxs: typing.Union[list, tuple],
    ) -> Group:
        if module not in self.module2node:
            raise ValueError(
                "Module {} is not in the dependency graph.".format(module)
            )
        # 考虑高阶的卷积情况
        if isinstance(module, ops.TORCH_CONV) and module.groups == module.out_channels and module.out_channels > 1:
            pruning_fn = function.prune_depthwise_conv_out_channels
        if isinstance(idxs, Number):
            idxs = [idxs]
        # self.update_index_mapping()
        # 创建组
        group = Group()
        root_node = self.module2node[module]
        group.add_dep(
            Dependency(pruning_fn, pruning_fn, source=root_node, target=root_node), idxs
        )
        visited_node = set()

        def _fix_dependency_graph_non_recursive(dep, idxs):
            # 获取顶层的处理栈，对dep的idxs相关通道进行剪枝
            processing_stk = [(dep, idxs)]
            while len(processing_stk) > 0:
                dep, idxs = processing_stk.pop(-1)
                node, fn = dep.target, dep.handler
                visited_node.add(node)
                for new_dep in node.dependencies:
                    # 如果剪枝函数fn会触发到对应的依赖
                    if new_dep.is_triggered_by(fn):
                        new_indices = idxs
                        # Todo: 依赖中的index_mapping是怎么赋值的？
                        for mapping in new_dep.index_mapping:
                            if mapping is not None:
                                # 这个mapping是个处理函数吗？
                                new_indices = mapping(new_indices)
                        if len(new_indices) == 0:
                            continue
                        # 已经添加过相关的依赖
                        if new_dep.target in visited_node and group.has_pruning_op(new_dep, new_indices):
                            continue
                        else:
                            # 添加到组中并继续之后的遍历
                            group.add_dep(new_dep, new_indices)
                            processing_stk.append((new_dep, new_indices))

        _fix_dependency_graph_non_recursive(*group[0])
        # 进行合并
        merged_group = Group()
        for dep, idxs in group.items:
            merged_group.add_and_merge(dep, idxs)
        merged_group._DG = self
        return merged_group

    def check_pruning_group(self, group: Group) -> bool:
        for dep, idxs in group:
            if self.is_out_channel_pruning_fn(dep.handler):
                prunable_chs = self.get_out_channels(dep.target.module)
                if prunable_chs is None: continue
                if prunable_chs <= len(idxs):
                    return False
            if self.is_in_channel_pruning_fn(dep.handler):
                prunable_in_chs = self.get_in_channels(dep.target.module)
                if prunable_in_chs is None: continue
                if prunable_in_chs <= len(idxs):
                    return False
        # 经验证，可以对指定的通道进行剪枝
        return True

    def get_out_channels(self, module_or_node):
        if isinstance(module_or_node, Node):
            module = module_or_node.module
            pruning_dim = module_or_node.pruning_dim
        else:
            module = module_or_node
            pruning_dim = self.module2node[module].pruning_dim
        p = self.get_pruner_of_module(module)
        p.pruning_dim = pruning_dim
        if p is None:
            return None
        return p.get_out_channels(module)

    def get_in_channels(self, module_or_node):
        if isinstance(module_or_node, Node):
            module = module_or_node.module
            pruning_dim = module_or_node.pruning_dim
        else:
            module = module_or_node
            pruning_dim = self.module2node[module].pruning_dim
        p = self.get_pruner_of_module(module)
        p.pruning_dim = pruning_dim
        if p is None:
            return None
        return p.get_in_channels(module)

    def is_out_channel_pruning_fn(self, fn: typing.Callable) -> bool:
        return fn in self._out_channel_pruning_fn

    def is_in_channel_pruning_fn(self, fn: typing.Callable) -> bool:
        return fn in self._in_channel_pruning_fn

    def get_all_groups(self, ignored_layers=[], root_module_types=(ops.TORCH_CONV, ops.TORCH_LINEAR)):
        visited_layers = []
        ignored_layers = ignored_layers + self.IGNORED_LAYERS
        for m in list(self.module2node.keys()):
            if m in ignored_layers:
                continue
            if not isinstance(m, tuple(root_module_types)):
                continue
            # 获取该模块m对应的剪枝器
            pruner = self.get_pruner_of_module(m)
            if pruner is None or pruner.get_out_channels(m) is None:
                continue
            if m in visited_layers:
                continue
            # 获取该层的输出通道
            layer_channels = pruner.get_out_channels(m)
            # 获取剪枝组
            # Todo: 注意，这里以对输出通道的剪枝作为访问过的组的依据
            #  对某一层的输出剪枝会影响到其他层的输入剪枝，如果其他层不存在层内依赖，则剪枝操作不会再进行传播
            #  因此，可以认为这里的其他层（的输出通道剪枝）未分组
            group = self.get_pruning_group(m, pruner.prune_out_channels, list(range(layer_channels)))
            prunable_group = True
            for dep, _ in group:
                module = dep.target.module
                pruning_fn = dep.handler
                if self.is_out_channel_pruning_fn(pruning_fn):
                    visited_layers.append(module)
                    if module in ignored_layers:
                        prunable_group = False
            if prunable_group:
                # 以生成器的形式返回创建的依赖组
                yield group


UnwrappedParameters = namedtuple('UnwrappedParameters', ['parameters', 'pruning_dim'])
