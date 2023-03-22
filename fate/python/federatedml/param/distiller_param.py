import json

from federatedml.param.base_param import BaseParam


class DistillerParam(BaseParam):
    def __init__(self,
                 secure_aggregate: bool = True,
                 aggregate_every_n_epoch: int = 1,
                 max_iter: int = 10,
                 batch_size: int = 20,
                 sched_dict: dict = None,
                 device: str = 'cpu',
                 pretrained: bool = True,
                 dataset: str = 'imagenet',
                 arch: str = 'alexnet',
                 lr: float = 0.005,
                 early_stop_eps: float = 0.,
                 post_train_quant: bool = False,
                 num_classes: int = 10,
                 quant_aware_aggregate: bool = False
                 ):
        super(DistillerParam, self).__init__()
        self.secure_aggregate = secure_aggregate
        self.aggregate_every_n_epoch = aggregate_every_n_epoch
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.sched_dict = sched_dict
        self.device = device
        self.pretrained = pretrained
        self.dataset = dataset
        self.arch = arch
        self.lr = lr
        self.early_stop_eps = early_stop_eps
        self.post_train_quant = post_train_quant
        self.num_classes = num_classes
        self.quant_aware_aggregate = quant_aware_aggregate

    def check(self):
        pass

    # Todo: 编写proto文件
    def generate_pb(self):
        from federatedml.protobuf.generated import distiller_meta_pb2
        pb = distiller_meta_pb2.DistillerParam()
        pb.secure_aggregate = self.secure_aggregate
        pb.aggregate_every_n_epoch = self.aggregate_every_n_epoch
        for sched_info in self.sched_dict:
            pb.sched_dict.append(json.dumps(sched_info))
        pb.batch_size = self.batch_size
        pb.max_iter = self.max_iter
        pb.num_classes = self.num_classes

        pb.device = self.device
        pb.pretrained = self.pretrained
        pb.dataset = self.dataset
        pb.arch = self.arch
        pb.lr = self.lr
        pb.early_stop_eps = self.early_stop_eps
        pb.post_train_quant = self.post_train_quant
        pb.quant_aware_aggregate = self.quant_aware_aggregate
        return pb

    def restore_from_pb(self, pb):
        self.secure_aggregate = pb.secure_aggregate
        self.aggregate_every_n_epoch = pb.aggregate_every_n_epoch
        for sched_info in pb.sched_dict:
            self.sched_dict.append(json.loads(sched_info))
        self.max_iter = pb.max_iter
        self.batch_size = pb.batch_size

        self.device = pb.device
        self.pretrained = pb.pretrained
        self.dataset = pb.dataset
        self.arch = pb.arch
        self.lr = pb.lr
        self.early_stop_eps = pb.early_stop_eps
        self.post_train_quant = pb.post_train_quant
        self.num_classes = pb.num_classes
        self.quant_aware_aggregate = pb.quant_aware_aggregate
        return pb
