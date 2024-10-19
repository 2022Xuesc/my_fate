from federatedml.nn.backend.utils.loader.dataset_loader import DatasetLoader
import torch
import math
category_dir = '/home/klaus125/research/fate/my_practice/dataset/coco' 
train_path = '/home/klaus125/research/fate/my_practice/dataset/coco/data/guest/train'
valid_path = '/home/klaus125/research/fate/my_practice/dataset/coco/data/guest/train'
inp_name = 'coco_glove_word2vec.pkl'

dataset_loader = DatasetLoader(category_dir, train_path, valid_path, inp_name=inp_name)

batch_size = 4
train_loader, valid_loader = dataset_loader.get_loaders(batch_size)
total_samples = len(train_loader.sampler)

steps = math.ceil(total_samples / batch_size)
metric = 0
for train_step, ((features, inp), target) in enumerate(train_loader):
    # target是一个4 * 80维的张量，统计该批次的方差
    metric += torch.mean(torch.var(target,dim=0)).item()
print(metric / steps)

# 不排序 0.035