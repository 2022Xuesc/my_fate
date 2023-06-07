import torch.utils.data
from torchvision.models import resnet18
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from federatedml.nn.backend.multi_label.models import *

import json
import os
import pickle

import cv2

category_dir = '/home/klaus125/research/fate/my_practice/dataset/coco'
#
images_dir = 'test_images'


def valid_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        # 输入图像是224*224
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class COCO(Dataset):
    def __init__(self, images_dir, config_dir, transforms=None, inp_name=None):
        self.images_dir = images_dir
        self.config_dir = config_dir
        self.transforms = transforms
        self.img_list = []
        self.cat2idx = None
        self.get_anno()

        self.num_classes = len(self.cat2idx)
        self.inp = None
        if inp_name is not None:
            inp_file = os.path.join(self.config_dir, inp_name)
            with open(inp_file, 'rb') as f:
                self.inp = pickle.load(f)
            self.inp_name = inp_name

    def get_anno(self):
        list_path = os.path.join(self.images_dir, 'anno.json')
        self.img_list = json.load(open(list_path, 'r'))
        category_path = os.path.join(self.config_dir, 'category.json')
        self.cat2idx = json.load(open(category_path, 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        img, target = self.get(item)
        # 如果self.inp不为空，说明是在GCN的配置环境下
        if self.inp is not None:
            return (img, self.inp), target
        else:  # 否则使用的是常规的网络，直接返回img和target即可
            return img, target

    def get(self, item):
        filename = item['file_name']
        labels = sorted(item['labels'])
        # 读取图像数据
        img = Image.open(os.path.join(self.images_dir, filename)).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        # Todo: 这里负标签设置为0，正标签设置为1
        target = np.zeros(self.num_classes, np.float32)
        target[labels] = 1
        return img, target


valid_dataset = COCO(images_dir, category_dir, valid_transforms())
valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset, batch_size=1
)

# 加载模型
model = SRN()
model.load_state_dict(torch.load('test_models/srn_model.pth'))
model = model.eval()
img_path = 'test_images/COCO_val2014_000000352694.jpg'

input_tensor = ...
for validate_step, (inputs, target) in enumerate(valid_loader):
    input_tensor = inputs

# Todo: 类激活图和attention 可视化有区别
img = read_image('test_images/COCO_val2014_000000352694.jpg')
# cam_extractor = SmoothGradCAMpp(model, target_layer=model.att_layer.conv3)
# out = model(input_tensor)
# activation_map = cam_extractor([58], out)
# result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# plt.imshow(result)
# plt.axis('off')
# plt.tight_layout()
# plt.show()


out = model(input_tensor)


attention_maps = numpy.load('stats/attention_maps.npy')

attention_mask = attention_maps[0][62]
print('Hello World')

# ratio = 0.5
# cmap = "jet"
# print("load image from: ", img_path)
# # load the image
# img = Image.open(img_path, mode='r')
# img_h, img_w = img.size[0], img.size[1]
# plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
#
# # scale the image
# img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
# img = img.resize((img_h, img_w))
# plt.imshow(img, alpha=1)
# plt.axis('off')
#
# mask = cv2.resize(attention_mask, (img_h, img_w))
# normed_mask = mask / mask.max()
# normed_mask = (normed_mask * 255).astype('uint8')
# plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
attention_mask = (attention_mask - attention_mask.min()) / (attention_mask.max() - attention_mask.min())

result = overlay_mask(to_pil_image(img), to_pil_image(attention_mask, mode='F'), alpha=0.5)
print(attention_mask)
plt.imshow(result)
plt.axis('off')
plt.tight_layout()
plt.show()
