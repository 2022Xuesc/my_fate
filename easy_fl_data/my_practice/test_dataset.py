import easyfl
import os
from torchvision import transforms

from easyfl.datasets import FederatedTorchDataset
from easyfl.datasets.dataset_util import *
from easyfl.distributed import slurm
from easyfl.models.full_salgl import FullSALGL

# 创建模型
import torchvision.models as torch_models

# python -m torch.distributed.run --nproc_per_node=6 --master_port=1234 test.py

import os
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])


config = {
    "gpu": 2,
    "distributed":{
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size
    }
}


category_dir = '/home/klaus125/research/EasyFL/easyfl/datasets/coco'
# Todo: 配置文件部分
train_paths = ['/home/klaus125/research/EasyFL/easyfl/datasets/coco/data/guest/train',
               '/home/klaus125/research/EasyFL/easyfl/datasets/coco/data/host/train']
val_paths = ['/home/klaus125/research/EasyFL/easyfl/datasets/coco/data/guest/val',
             '/home/klaus125/research/EasyFL/easyfl/datasets/coco/data/host/val']
client_ids = [0, 1]

inp_name = 'coco_glove_word2vec.pkl'
is_gcn = inp_name is not None

resize_scale = 512
crop_scale = 448

train_datasets = {}
val_datasets = {}
for i in range(len(client_ids)):
    train_dataset = COCO(
        images_dir=train_paths[i],
        config_dir=category_dir,
        transforms=train_transforms(resize_scale, crop_scale, is_gcn),
        inp_name=inp_name
    )
    train_datasets[client_ids[i]] = train_dataset

    # Todo: 设定验证数据集
    val_dataset = COCO(
        images_dir=val_paths[i],
        config_dir=category_dir,
        transforms=valid_transforms(resize_scale, crop_scale, is_gcn),
        inp_name=inp_name
    )
    val_datasets[client_ids[i]] = val_dataset

train_data = FederatedTorchDataset(train_datasets, client_ids)
val_data = FederatedTorchDataset(val_datasets, client_ids)

easyfl.register_dataset(train_data, val_data)

model = torch_models.resnet101(pretrained=True, num_classes=1000)
model = FullSALGL(model)
easyfl.register_model(model)


easyfl.init(config)

easyfl.run()