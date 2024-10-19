import os 
import json
dir = '/home/klaus125/research/fate/my_practice/dataset/coco/data/guest/train'
files = os.listdir(dir)
images = []
anno_file = None
for file in files:
    if file.endswith('json'):
        anno_file = file
    elif file.endswith('.jpg'):
        images.append(file)
annotations = json.load(open(os.path.join(dir,anno_file),'r'))
# 维护一个dict()，每个标签对应一个图像列表
map = dict()
# 遍历每一个图像
for image in images:
    # 根据名称获取标签
    annotations[image]