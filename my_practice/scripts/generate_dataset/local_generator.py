from dataset_generator import *

src_dir = '../dataset/ms-coco/src_val'
guest_train_dir = '../dataset/ms-coco/guest/train'
host_train_dir = '../dataset/ms-coco/host/train'

guest_val_dir = '../dataset/ms-coco/guest/val'
host_val_dir = '../dataset/ms-coco/host/val'
dirs = [guest_train_dir, guest_val_dir, host_train_dir, host_val_dir]


# 共40000张图片，每个客户端训练2000张，测试200张


generate_2014(src_dir, 0, 0.05, guest_train_dir)
generate_2014(src_dir, 0.05, 0.1, host_train_dir)
#
generate_2014(src_dir, 0.1, 0.105, guest_val_dir)
generate_2014(src_dir, 0.105, 0.11, host_val_dir)
#
#
# generate_labels(dirs)
#
# generate_configs(dirs)