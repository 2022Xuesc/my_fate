import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from time import time
import multiprocessing as mp


class MultiLabelDataSet:
    def __init__(self, images_dir, transform=None):
        super(MultiLabelDataSet, self).__init__()
        label_path = os.path.join(images_dir, 'labels.txt')
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
            labels.append([float(l) for l in info[1:len(info)]])
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


train_path = '/data/projects/my_dataset/client1/train'
# train_path = '/home/klaus125/research/dataset/ms-coco/guest/train'
train_transforms = transforms.Compose([
    # 将图像缩放为256*256
    transforms.Resize(256),
    # 随机裁剪出224*224大小的图像用于训练
    transforms.RandomResizedCrop(224),
    # 将图像进行水平翻转
    transforms.RandomHorizontalFlip(),
    # 转换为张量
    transforms.ToTensor(),
    # 对图像进行归一化，以下两个list分别是RGB通道的均值和标准差
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_data_set = MultiLabelDataSet(train_path,
                                   train_transforms)

print(f"num of CPU: {mp.cpu_count()}")
device = 'cuda:5'
batch_size = 128

for num_workers in range(0, mp.cpu_count(), 2):
    for pin in range(0, 2):
        train_loader = torch.utils.data.DataLoader(train_data_set, shuffle=True, num_workers=num_workers,
                                                   batch_size=batch_size, pin_memory=(pin == 0))
        start = time()
        for epoch in range(1, 3):
            for i, (inputs, target) in enumerate(train_loader, 0):
                inputs = inputs.to(device)
                target = target.to(device)
        end = time()
        print("Finish with:{} second, num_workers={},pin_memory = {}".format(end - start, num_workers, pin == 0))
