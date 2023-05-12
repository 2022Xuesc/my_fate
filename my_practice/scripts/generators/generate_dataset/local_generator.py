from dataset_generator import *

dataset_dir = '/home/klaus125/research/dataset/'

src_dir = os.path.join(dataset_dir, 'val2014')
target_dir = "../../../dataset/coco/data"
guest_train_dir = os.path.join(target_dir, 'guest/train')
host_train_dir = os.path.join(target_dir, 'host/train')

guest_val_dir = os.path.join(target_dir, 'guest/val')
host_val_dir = os.path.join(target_dir, 'host/val')

dirs = [guest_train_dir, guest_val_dir, host_train_dir, host_val_dir]

# 共40000张图片，每个客户端训练2000张，测试200张


generate_2014(src_dir, 0, 0.0005, guest_train_dir)
generate_2014(src_dir, 0.0005, 0.001, host_train_dir)

generate_2014(src_dir, 0.001, 0.00105, guest_val_dir)
generate_2014(src_dir, 0.00105, 0.0011, host_val_dir)
#
#

coco_dir = '../../../dataset/coco'
for dir in dirs:
    generate_anno(coco_dir, dir)
#
# generate_configs(dirs)
