import torch.optim

from easyfl.client.base import BaseClient

import time
from torch import nn
import torch.optim as optim
from easyfl.client.base import BaseClient
from easyfl.losses.AsymmetricLoss import AsymmetricLossOptimized
from easyfl.metrics.APMeter import AveragePrecisionMeter


# Inherit BaseClient to implement customized client operations.
class MyClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(MyClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        self.ap_meter = AveragePrecisionMeter(difficult_examples=False)
        self.epoch_scene_cnts = None

    def load_loss_fn(self, conf):
        # 返回损失函数
        return AsymmetricLossOptimized()

    def load_optimizer(self, conf):
        lr, lrp = conf.lr, 1
        optimizer = optim.SGD(
            self.model.get_config_optim(lr=lr, lrp=lrp),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4)
        return optimizer

    def upload(self):
        pass

    def train(self, conf, device='cpu'):
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        self.train_loss = []
        for i in range(conf.local_epoch):
            self.ap_meter.reset()
            batch_loss = []
            for batched_x, batched_y in self.train_loader:
                x, y = batched_x.to(device), batched_y.to(device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
    
    def train_one_epoch(self,epoch):
        self.ap_meter.reset()