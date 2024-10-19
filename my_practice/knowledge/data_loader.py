import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os

from PIL import Image


class MyDataset(data.Dataset):
    def __init__(self, label_path, images_dir, transform=None):
        super(MyDataset, self).__init__()
        # 打开存储图像名称与标签的txt文件
        fp = open(label_path, 'r')
        images = []
        labels = []
        for line in fp:
            # 移除首位的回车符
            line.strip('\n')
            # 移除末尾的空格符
            line.rstrip()
            info = line.split(',')
            images.append(info[0])
            # 将标签信息转为float类型
            labels.append([int(l) for l in info[1:len(info)]])
        self.images = images
        self.images_dir = images_dir
        self.labels = labels
        self.transform = transform

    # 重写该函数进行图像数据的读取
    # 通过索引的访问方式
    def __getitem__(self, item):
        image_name = self.images[item]
        # 这里的label是一个list列表，表示标签向量
        label = self.labels[item]
        # 从loader中根据图像名称读取图像信息
        image = Image.open(os.path.join(self.images_dir, image_name)).convert('RGB')
        # 处理图像数据
        if self.transform is not None:
            image = self.transform(image)
        # 将浮点数的list转换为float tensor
        label = torch.FloatTensor(label)
        # 返回处理后的内容
        return image, label

    def __len__(self):
        return len(self.images)


def collate_fn(data):
    # 根据label长度对data进行降序排序
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, labels = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(label) for label in labels]
    targets = torch.zeros(len(labels), max(lengths)).long()

    for i, label in enumerate(labels):
        end = lengths[i]
        targets[i, :end] = label[:end]
    return images, targets, lengths


def get_loader(label_path, images_dir, transform, batch_size, shuffle, num_workers):
    coco = MyDataset(label_path, images_dir, transform)
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
