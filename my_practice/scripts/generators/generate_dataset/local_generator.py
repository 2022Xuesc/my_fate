from dataset_generator import *

# home/klaus125/research/dataset/NUS-WIDE/images
dataset_dir = '/home/klaus125/research/dataset/NUS-WIDE/images'

src_dir = os.path.join(dataset_dir, 'val')
target_dir = "../../../dataset/nuswide/data"
guest_train_dir = os.path.join(target_dir, 'guest/train')
host_train_dir = os.path.join(target_dir, 'host/train')

guest_val_dir = os.path.join(target_dir, 'guest/val')
host_val_dir = os.path.join(target_dir, 'host/val')

dirs = [guest_train_dir, guest_val_dir, host_train_dir, host_val_dir]
# dirs = ['/home/klaus125/research/dataset/val2014']
# 共40000张图片，每个客户端训练2000张，测试200张


# generate_2014(src_dir, 0, 0.0001, guest_train_dir)
# generate_2014(src_dir, 0.0001, 0.0002, host_train_dir)

# generate_2014(src_dir, 0, 0.0001, guest_val_dir)
# generate_2014(src_dir, 0.0001, 0.0002, host_val_dir)
#
#

# coco_dir = '../../../dataset/coco'
# for dir in dirs:
#     generate_anno(coco_dir, dir)
# generate_anno(coco_dir,'/home/klaus125/research/fate/my_practice/dataset/coco/data/guest/train')
#
# generate_configs(dirs)
dataset = "nuswide"
data_dir = '../../../dataset/nuswide'
generate_anno(data_dir, host_train_dir, dataset, phase='train')
generate_anno(data_dir, guest_train_dir, dataset, phase='train')
generate_anno(data_dir, host_val_dir, dataset, phase='val')
generate_anno(data_dir, guest_val_dir, dataset, phase='val')
