import os
import shutil

base_path = '/data/projects/data.imagenet'
#base_path = '/home/klaus125/research/data.imagenet/'
src_path = os.path.join(base_path, 'src')
src_100_100_path = os.path.join(base_path, 'src_100_100')


def check(dir_list):
    for dir_path in dir_list:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def do_extract():
    dirnames = os.listdir(src_path)
    for i in range(100):
        print(os.path.join(src_path,dirnames[i]))
        shutil.copytree(os.path.join(src_path, dirnames[i]), os.path.join(src_100_100_path,dirnames[i]))


if __name__ == '__main__':
    # check([src_100_100_path])
    do_extract()
