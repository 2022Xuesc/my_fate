import os
import shutil

base_path = '/home/klaus125/research/data.imagenet'
src_path = os.path.join(base_path, 'src')
src_100_100_path = os.path.join(base_path, 'src_10_100')


def do_extract():
    dirnames = os.listdir(src_path)
    for i in range(10):
        print(os.path.join(src_path,dirnames[i]))
        shutil.copytree(os.path.join(src_path, dirnames[i]), os.path.join(src_100_100_path,dirnames[i]))


if __name__ == '__main__':
    do_extract()
