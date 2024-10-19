from federatedml.framework.homo.blocks.base import HomoTransferBase
from federatedml.framework.homo.blocks.has_converged import HasConvergedTransVar
from federatedml.framework.homo.blocks.loss_scatter import LossScatterTransVar
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from federatedml.model_base import ModelBase, MetricMeta, Metric
from federatedml.param.gcn_param import GCNParam
from federatedml.util import consts


class GCNBase(ModelBase):
    # Todo: 哪里定义传输变量了？进行debug
    #  复用已编写模块的传输变量
    def __init__(self, trans_var):
        super().__init__()
        # 指定模块参数类型为自定义的参数类
        self.model_param = GCNParam()
        self.transfer_variable = trans_var

    def _init_model(self, param):
        self.param = param


class GCNServer(GCNBase):
    def __init__(self, trans_var):
        super().__init__(trans_var=trans_var)
        self.aggregator = ...
        self._init_iteration = 0

    def _init_model(self, param: GCNParam):
        super()._init_model(param)

    # 回调损失
    def callback_loss(self, iter_num, loss):
        metric_meta = MetricMeta(
            name="train",
            metric_type="LOSS",
            extra_metas={
                "unit_name": "iters",
            },
        )

        self.callback_meta(
            metric_name="loss", metric_namespace="train", metric_meta=metric_meta
        )
        self.callback_metric(
            metric_name="loss",
            metric_namespace="train",
            metric_data=[Metric(iter_num, loss)],
        )

    # 定义服务器端的拟合逻辑
    def fit(self, train_data, valid_data):
        # Todo: 服务器端
        # self.callback_list.on_train_begin({train_data, valid_data}, None)
        from federatedml.nn.gcn._fate import build_aggregator
        self.aggregator = build_aggregator(self.param, init_iteration=self._init_iteration)
        # 数据集对齐
        # self.aggregator.dataset_align()
        # 进行拟合
        self.aggregator.fit(self.callback_loss)
        # 回调
        self.callback_list.on_train_end()


class GCNClient(GCNBase):
    def __init__(self, trans_var):
        super().__init__(trans_var=trans_var)
        self._fitter = ...

    def _init_model(self, param: GCNParam):
        super()._init_model(param)

    # Todo: 查看train_data的格式
    def fit(self, train_data, valid_data):
        from federatedml.nn.gcn._fate import build_fitter
        self._fitter = None
        self._fitter, train_loader, valid_loader, agg_type = build_fitter(
            param=self.param,
            train_data=train_data,
            valid_data=valid_data
        )
        self._fitter.fit(train_loader, valid_loader, agg_type)
        self.callback_list.on_train_end()


class GCNDefaultTransVar(HomoTransferBase):
    def __init__(
            self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None
    ):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.secure_aggregator_trans_var = SecureAggregatorTransVar(
            server=server, clients=clients, prefix=self.prefix
        )
        self.loss_scatter_trans_var = LossScatterTransVar(
            server=server, clients=clients, prefix=self.prefix
        )
        self.has_converged_trans_var = HasConvergedTransVar(
            server=server, clients=clients, prefix=self.prefix
        )


class GCNDefaultServer(GCNServer):
    def __init__(self):
        super().__init__(trans_var=GCNDefaultTransVar)


class GCNDefaultClient(GCNClient):
    def __init__(self):
        super().__init__(trans_var=GCNDefaultTransVar())
