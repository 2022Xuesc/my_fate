import torch
import torch.nn as nn
import typing
import federatedml.nn.backend.multi_label.prunners.depgraph.ops as ops
import federatedml.nn.backend.multi_label.prunners.depgraph.function as function
import federatedml.nn.backend.multi_label.prunners.depgraph.dependency as dependency


def linear_scheduler(ch_sparsity_dict, steps):
    return [((i) / float(steps)) * ch_sparsity_dict for i in range(steps + 1)]



# 元剪枝器的实现
class MetaPruner:
    def __init__(
            self,
            model: nn.Module,
            example_inputs: torch.Tensor,
            # 涉及到参数组的重要性度量
            importance: typing.Callable,
            # Todo: global_pruning涉及到组间剪枝？
            global_pruning: bool = False,
            ch_sparsity: float = 0.5,
            ch_sparsity_dict: typing.Dict[nn.Module, float] = None,
            max_ch_sparsity: float = 1.0,
            iterative_steps: int = 1,
            iterative_sparsity_scheduler: typing.Callable = linear_scheduler,
            ignored_layers: typing.List[nn.Module] = None,
            # Advanced
            round_to: int = None,  # round channels to 8x, 16x, ...
            # for grouped channels.
            channel_groups: typing.Dict[nn.Module, int] = dict(),
            # pruners for customized layers
            customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None,
            # unwrapped nn.Parameters like ViT.pos_emb
            unwrapped_parameters: typing.List[nn.Parameter] = None,
            root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],
            # root module for each group
            output_transform: typing.Callable = None,
    ):
        self.model = model
        self.importance = importance
        self.ch_sparsity = ch_sparsity
        self.ch_sparsity_dict = ch_sparsity_dict if ch_sparsity_dict is not None else {}
        # 最大通道稀疏度
        self.max_ch_sparsity = max_ch_sparsity
        self.global_pruning = global_pruning
        self.channel_groups = channel_groups
        self.root_module_types = root_module_types
        self.round_to = round_to

        # 先构建依赖图
        self.DG = dependency.DependencyGraph().build_dependency(
            model,
            example_inputs,
            output_transform,
            unwrapped_parameters,
            customized_pruners
        )
        self.ignored_layers = []
        # 配置剪枝忽略的层
        if ignored_layers:
            for layer in ignored_layers:
                self.ignored_layers.extend(list(layer.modules))
        self.iterative_steps = iterative_steps
        self.iterative_sparsity_scheduler = iterative_sparsity_scheduler
        self.current_step = 0

        self.layer_init_out_ch = {}
        self.layer_init_in_ch = {}
        for m in self.DG.module2node.keys():
            if ops.module2type(m) in self.DG.REGISTERED_PRUNERS:
                self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
                self.layer_init_in_ch[m] = self.DG.get_in_channels(m)
        # 每个迭代步的全局通道稀疏率
        self.per_step_ch_sparsity = self.iterative_sparsity_scheduler(
            self.ch_sparsity, self.iterative_steps
        )
        # 暂时跳过自定义的稀疏率
        if self.ch_sparsity_dict is not None:
            pass
        # 暂时跳过group相关的设置
        # 从网络结构中获取group的相关设置
        if self.global_pruning:
            initial_total_channels = 0
            # 计算网络中所有组的总通道数
            # Todo: 需要注意的是单组内的通道数是一致的
            for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers,
                                                root_module_types=self.root_module_types):
                ch_groups = self.get_channel_groups()
                # 对于通道已经分组的情况，剪枝需要在组上进行，因此，通道数应该除以通道组的大小
                initial_total_channels += self.DG.get_out_channels(group[0][0].target.module) // ch_groups
            self.initial_total_channels = initial_total_channels

    def pruning_history(self):
        return self.DG.pruning_history()

    def load_pruning_history(self, pruning_history):
        self.DG.load_pruning_history(pruning_history)

    def get_target_sparsity(self, module):
        # 获取对应的模块的稀疏率配置，如果无特定配置，则使用当前步通用的通道稀疏率
        # 返回前需要受到最大通道稀疏率的制约
        s = self.ch_sparsity_dict.get(module, self.per_step_ch_sparsity)[self.current_step]
        return min(s, self.max_ch_sparsity)

    def reset(self):
        self.current_step = 0

    # 模型正则化器
    def regularize(self, model, loss):
        pass

    # 步进剪枝
    # 如果设定交互式，则以yield方式返回剪枝结果
    # 否则，直接执行剪枝
    def step(self,select_all=True, interactive=False):
        name2masks = {}
        # 进行实际的剪枝操作
        for name, p in self.model.named_parameters():
            # 建立从模块名称到mask矩阵的映射
            name2masks[name] = torch.ones_like(p)

        self.current_step += 1
        if self.global_pruning:
            if interactive:
                return self.prune_global()
            else:
                # 对于每个组，生成剪枝索引，并进行剪枝
                for group in self.prune_global():
                    group.prune(name2masks)
        else:
            if interactive:
                return self.prune_local()
            else:
                # Todo: 对于每个组，进行剪枝
                for group in self.prune_local(select_all):
                    group.prune(name2masks)
        # print(name2masks)
        num_zeros = 0
        num_ones = 0

        for tensor in list(name2masks.values()):
            num_zeros += (tensor == 0).sum().item()
            num_ones += (tensor == 1).sum().item()

        # print("Number of zeros:", num_zeros)
        # print("Number of ones:", num_ones)
        # print("Sparsity: ", num_zeros / (num_zeros + num_ones))
        return name2masks

    # Todo: 两个重要的剪枝函数 --> 局部剪枝和全局剪枝

    def prune_local(self,select_all=True):
        if self.current_step > self.iterative_steps:
            return
        # 每个组分别考虑
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers,
                                            root_module_types=self.root_module_types):
            # 提取层名
            select = {'layer2', 'layer1', 'conv1'}
            group_name = group.items[0].dep.target._name
            first_dot_index = group_name.find(".")
            layer_name = None
            if first_dot_index != -1:
                layer_name = group_name[:first_dot_index]
            else:
                layer_name = group_name
            if not select_all and layer_name not in select:
                continue
            if self._check_sparsity(group):
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler
                ch_groups = self.get_channel_groups()
                imp = self.estimate_importance(group)
                if imp is None:
                    continue
                current_channels = self.DG.get_out_channels(module)
                target_sparsity = self.get_target_sparsity(module)
                n_pruned = current_channels - int(
                    self.layer_init_out_ch[module] *
                    (1 - target_sparsity)
                )

                imp_argsort = torch.argsort(imp)
                pruning_idxs = imp_argsort[:(n_pruned // ch_groups)]

                group = self.DG.get_pruning_group(module, pruning_fn, pruning_idxs.tolist())
                if self.DG.check_pruning_group(group):
                    yield group

    def prune_global(self):
        if self.current_step > self.iterative_steps:
            return
        # 计算每个组的重要性
        global_importance = []

        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers,
                                            root_module_types=self.root_module_types):
            if self._check_sparsity(group):
                ch_groups = self.get_channel_groups()
                # 估计该参数组group的重要性
                imp = self.estimate_importance(group, ch_groups=ch_groups)
                if imp is None: continue
                if ch_groups > 1:
                    imp = imp[:len(imp) // ch_groups]
                # 添加组的重要性
                global_importance.append((group, ch_groups, imp))

        if len(global_importance) == 0:
            return
        # Todo: 这里的重要性是每个组的重要性还是每个组+每个通道的重要性？
        #  Answer: 从代码上看，应该是每个通道的重要性
        imp = torch.cat([local_imp[-1] for local_imp in global_importance], dim=0)
        # 获取当前步的通道稀疏率
        target_sparsity = self.per_step_ch_sparsity[self.current_step]
        # 计算需要剪枝的通道数
        n_pruned = len(imp) - int(self.initial_total_channels * (1 - target_sparsity))
        if n_pruned <= 0:
            return
        topk_imp, _ = torch.topk(imp, k=n_pruned, largest=False)
        threshold = topk_imp[-1]

        for group, ch_groups, imp in global_importance:
            module = group[0][0].target.module
            pruning_fn = group[0][0].handler
            # 确定剪枝索引-->此时mask就已经确定了；由于非按序遍历，因此，维护模块名到mask的字典，最后再重新组装即可
            pruning_indices = (imp <= threshold).nonzero().view(-1)
            # Todo: 暂时跳过对ch_groups大于1的情况的处理
            # if ch_groups > 1:
            #     group_size = self.DG.get_out_channels(module) // ch_groups
            #     pruning_indices = torch.cat([pruning_indices + group_size * i for i in range(ch_groups)], 0)
            # Todo: 跳过对round_to的配置
            # if self.round_to:
            #     n_pruned = len(pruning_indices)
            #     n_pruned = n_pruned - (n_pruned % self.round_to)
            #     pruning_indices = pruning_indices[:n_pruned]
            group = self.DG.get_pruning_group(module, pruning_fn, pruning_indices.tolist())
            # 剪枝符合要求
            # Todo: 这里不是真的需要进行剪枝操作，而是遍历group，为指定模块生成mask矩阵即可
            if self.DG.check_pruning_group(group):
                yield group

    def estimate_importance(self, group, ch_groups=1):
        # 调用初始化时指定的重要性度量函数
        return self.importance(group, ch_groups=ch_groups)

    def get_channel_groups(self):
        return 1

    def _check_sparsity(self, group):
        # 对于每个依赖dep
        for dep, _ in group:
            module = dep.target.module
            pruning_fn = dep.handler
            if dep.target.type == ops.OPTYPE.PARAMETER:
                continue
            # 如果剪枝函数是输出通道的剪枝函数
            if self.DG.is_out_channel_pruning_fn(pruning_fn):
                # 获取模块的目标稀疏率
                target_sparsity = self.get_target_sparsity(module)
                # 获取层输出通道的数目
                layer_out_ch = self.DG.get_out_channels(module)
                if layer_out_ch is None: continue
                # 如果层的输出通道已经足够小了，直接返回False，表明不允许剪枝
                if layer_out_ch < self.layer_init_out_ch[module] * (1 - self.max_ch_sparsity) or layer_out_ch == 1:
                    return False
            # 如果剪枝函数是输入通道的剪枝函数
            elif self.DG.is_in_channel_pruning_fn(pruning_fn):
                # 获取输入通道
                layer_in_ch = self.DG.get_in_channels(module)
                if layer_in_ch is None: continue
                # 如果该层的输入通道数已经足够小了，直接返回False，表明不允许剪枝
                if layer_in_ch < self.layer_init_in_ch[module] * (1 - self.max_ch_sparsity) or layer_in_ch == 1:
                    return False
        return True
