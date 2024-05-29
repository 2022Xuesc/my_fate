import torchvision.transforms as transforms

from federatedml.nn.backend.gcn.utils import *
from federatedml.nn.backend.pytorch.data import COCO
from federatedml.nn.backend.pytorch.data import VOC


def gcn_train_transforms(resize_scale, crop_scale):
    return transforms.Compose([
        # 将短边缩放到resize_scale，另一边等比例缩放
        transforms.Resize((resize_scale, resize_scale)),
        MultiScaleCrop(crop_scale, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def gcn_valid_transforms(resize_scale, crop_scale):
    return transforms.Compose([
        Warp(crop_scale),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def train_transforms(resize_scale, crop_scale, is_gcn=False):
    if is_gcn:
        return gcn_train_transforms(resize_scale, crop_scale)
    return transforms.Compose([
        # 将图像缩放为256*256
        transforms.Resize(resize_scale),
        # 随机裁剪出224*224大小的图像用于训练
        transforms.RandomResizedCrop(crop_scale),
        # 将图像进行水平翻转
        transforms.RandomHorizontalFlip(),
        # 转换为张量
        transforms.ToTensor(),
        # 对图像进行归一化，以下两个list分别是RGB通道的均值和标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def valid_transforms(resize_scale, crop_scale, is_gcn=False):
    if is_gcn:
        return gcn_valid_transforms(resize_scale, crop_scale)
    return transforms.Compose([
        transforms.Resize(resize_scale),
        # 输入图像是224*224
        transforms.CenterCrop(crop_scale),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class DatasetLoader(object):
    # category_dir是配置文件所在的目录
    def __init__(self, category_dir, train_path=None, valid_path=None, inp_name=None):
        super(DatasetLoader, self).__init__()
        self.category_dir = category_dir
        self.train_path = train_path
        self.valid_path = valid_path
        self.inp_name = inp_name
        self.is_gcn = inp_name is not None

    # 传resize_scale，一般是512或256
    # 传crop_scale，一般是448或224
    def get_loaders(self, batch_size, resize_scale=512, crop_scale=448, dataset='COCO', shuffle=True, drop_last=True):
        if dataset == 'COCO':
            train_dataset = COCO(images_dir=self.train_path,
                                 config_dir=self.category_dir,
                                 transforms=train_transforms(resize_scale, crop_scale, is_gcn=self.is_gcn),
                                 inp_name=self.inp_name)
            valid_dataset = COCO(images_dir=self.valid_path,
                                 config_dir=self.category_dir,
                                 transforms=valid_transforms(resize_scale, crop_scale, is_gcn=self.is_gcn),
                                 inp_name=self.inp_name)
        else:
            train_dataset = VOC(images_dir=self.train_path,
                                config_dir=self.category_dir,
                                transforms=train_transforms(resize_scale, crop_scale, is_gcn=self.is_gcn),
                                inp_name=self.inp_name)
            valid_dataset = VOC(images_dir=self.valid_path,
                                config_dir=self.category_dir,
                                transforms=valid_transforms(resize_scale, crop_scale, is_gcn=self.is_gcn),
                                inp_name=self.inp_name)
        # 对batch_size进行修正
        batch_size = max(1, min(batch_size, len(train_dataset), len(valid_dataset)))

        num_workers = 32

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
            drop_last=drop_last, shuffle=shuffle
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset, batch_size=batch_size, num_workers=num_workers,
            drop_last=drop_last, shuffle=False
        )
        return train_loader, valid_loader
