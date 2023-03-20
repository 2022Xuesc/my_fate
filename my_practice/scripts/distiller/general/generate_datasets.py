import math
import numpy as np

import json
import os
import shutil

base_path = '/home/klaus125/research/data.imagenet'
src_path = base_path + '/src_10_10'
iid_path = base_path + '/iid10'
non_iid_path = base_path + '/non-iid'


def generate_iid_src():
    host_dst = iid_path + "/host/src"
    guest_dst = iid_path + "/guest/src"
    # 判断目录是否存在，如果不存在，则进行创建
    check([host_dst, guest_dst])

    # 读取src下的所有文件夹
    dirnames = os.listdir(src_path)
    for dirname in dirnames:
        dir_path = src_path + "/" + dirname
        filenames = os.listdir(dir_path)
        num_samples = len(filenames)
        # 将前一半移动到iid/host/src文件夹下
        for i in range(num_samples // 2):
            file = dir_path + '/' + filenames[i]
            shutil.copy(file, host_dst)
        for i in range(num_samples // 2, num_samples):
            file = dir_path + '/' + filenames[i]
            shutil.copy(file, guest_dst)


# 暂时认为每个客户端拥有50%的类别数，并且拥有每个类别下全部的训练样本
def generate_non_iid_src():
    host_dst = non_iid_path + "/host/src"
    guest_dst = non_iid_path + "/guest/src"
    check([host_dst, guest_dst])
    dirnames = os.listdir(src_path)
    num_dirs = len(dirnames)
    for i in range(num_dirs // 2):
        dir_path = src_path + "/" + dirnames[i]
        # 将该目录下的所有图片都移动到non-iid/host/src文件夹下
        for filename in os.listdir(dir_path):
            shutil.copy(dir_path + "/" + filename, host_dst)
    for i in range(num_dirs // 2, num_dirs):
        dir_path = src_path + "/" + dirnames[i]
        # 将该目录下的所有图片都移动到non-iid/guest/src文件夹下
        for filename in os.listdir(dir_path):
            shutil.copy(dir_path + "/" + filename, guest_dst)


def split_valid_test(path, valid_ratio, test_ratio):
    guest_path = path + "/guest"
    host_path = path + "/host"
    do_split(guest_path, valid_ratio, test_ratio)
    do_split(host_path, valid_ratio, test_ratio)

    # 删除src目录
    deldir(os.path.join(guest_path, 'src'))
    deldir(os.path.join(host_path, 'src'))


def do_split(role_path, valid_ratio, test_ratio):
    # 检查目标路径是否存在
    train_path, valid_path, test_path = role_path + "/train", role_path + "/valid", role_path + "/test"
    check([train_path, valid_path, test_path])
    # 枚举role_path/src下的所有文件
    image_path = role_path + "/src"
    filenames = os.listdir(image_path)
    num_samples = len(filenames)
    valid_size = math.floor(valid_ratio * num_samples)
    test_size = math.floor(test_ratio * num_samples)
    train_size = num_samples - valid_size - test_size

    check([train_path + "/images", valid_path + "/images", test_path + "/images"])
    for i in range(train_size):
        shutil.copy(image_path + "/" + filenames[i], train_path + "/images")
    for i in range(train_size, train_size + valid_size):
        shutil.copy(image_path + "/" + filenames[i], valid_path + "/images")
    for i in range(train_size + valid_size, num_samples):
        shutil.copy(image_path + "/" + filenames[i], test_path + "/images")


def check(dir_list):
    for dir_path in dir_list:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def deldir(dirname):
    if not os.path.exists(dirname):
        return False
    if os.path.isfile(dirname):
        os.remove(dirname)
        return
    for i in os.listdir(dirname):
        t = os.path.join(dirname, i)
        if os.path.isdir(t):
            deldir(t)  # 重新调用次方法
        else:
            os.unlink(t)
    os.removedirs(dirname)  # 递归删除目录下面的空文件夹


def fatelize(path):
    do_fatelize(path + '/guest/train')
    do_fatelize(path + '/guest/valid')
    do_fatelize(path + '/guest/test')
    do_fatelize(path + '/host/train')
    do_fatelize(path + '/host/valid')
    do_fatelize(path + '/host/test')


def do_fatelize(data_path):
    # data_path目录下的文件移动到images目录中，根据文件名生成filenames文件和targets文件
    images_path = data_path + "/images"
    filenames = os.listdir(images_path)
    for i in range(len(filenames)):
        filename = filenames[i]
        # # 给文件加前缀，使其按添加顺序排列
        old_name = filename
        new_name = str(i) + "_" + filename
        os.rename(images_path + "/" + old_name, images_path + "/" + new_name)

        filename_without_ext = filename[:-5]
        class_name = filename_without_ext.split('_')[0]
        target = class_dict[class_name]
        # 创建filenames文件
        filenames_file = open(data_path + '/filenames', 'a')
        filenames_file.write(str(i) + '_' + filename_without_ext + '\n')
        # 创建targets文件
        targets_file = open(data_path + '/targets', 'a')
        targets_file.write(str(i) + '_' + filename_without_ext + "," + str(target) + '\n')


def save_dict():
    index = 0
    dirnames = os.listdir(src_path)
    for dirname in dirnames:
        dir_path = src_path + "/" + dirname
        for file in os.listdir(dir_path):
            classname = file[:-5].split('_')[0]
            if classname not in class_dict:
                class_dict[classname] = index
                index += 1
    json_obj = json.dumps(class_dict)
    file_obj = open('class_dict.json', 'w')
    file_obj.write(json_obj)
    file_obj.close()


def load_dict():
    with open('class_dict.json', 'r') as f:
        clz_dict = json.load(f)
        return clz_dict


class_dict = dict()

if __name__ == '__main__':
    save_dict()
    class_dict = load_dict()
    generate_iid_src()
    valid_ratio, test_ratio = 0.2, 0.1
    split_valid_test(iid_path, valid_ratio, test_ratio)
    fatelize(iid_path)


    # 配置non-iid的数据集
    # generate_non_iid_src()
    # split_valid_test(non_iid_path, valid_ratio, test_ratio)
    # fatelize(non_iid_path)
    # print('hello there')
