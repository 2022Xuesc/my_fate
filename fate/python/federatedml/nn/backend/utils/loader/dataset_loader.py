from federatedml.nn.backend.pytorch.data import COCO
import torch
import torchvision.transforms as transforms


def train_transforms(resize_scale, crop_scale):
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


def valid_transforms(resize_scale, crop_scale):
    return transforms.Compose([
        transforms.Resize(resize_scale),
        # 输入图像是224*224
        transforms.CenterCrop(crop_scale),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class DatasetLoader(object):
    # category_dir是配置文件所在的目录
    def __init__(self, category_dir, train_path, valid_path):
        super(DatasetLoader, self).__init__()
        self.category_dir = category_dir
        self.train_path = train_path
        self.valid_path = valid_path

    # 传resize_scale，一般是512或256
    # 传crop_scale，一般是448或224
    def get_loaders(self, batch_size, resize_scale=512, crop_scale=448):
        train_dataset = COCO(images_dir=self.train_path,
                             config_dir=self.category_dir,
                             transforms=train_transforms(resize_scale, crop_scale))
        valid_dataset = COCO(images_dir=self.valid_path,
                             config_dir=self.category_dir,
                             transforms=train_transforms(resize_scale, crop_scale))

        # 对batch_size进行修正
        batch_size = max(1, min(batch_size, len(train_dataset), len(valid_dataset)))

        shuffle = True
        drop_last = False
        num_workers = 32

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
            drop_last=drop_last, shuffle=shuffle
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset, batch_size=batch_size, num_workers=num_workers,
            drop_last=drop_last, shuffle=shuffle
        )
        return train_loader, valid_loader
