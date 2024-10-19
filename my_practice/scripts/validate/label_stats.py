import json

anno_list = []

for i in range(1, 11):
    anno_list.append(f'/data/projects/iid_dataset/client{i}/train/anno.json')
num_labels = 80

for anno_file in anno_list:
    anno = json.load(open(anno_file, 'r'))
    total_labels = set()
    # 对于每一张图片的信息,统计其出现次数
    for info in anno:
        for label in info['labels']:
            total_labels.add(label)
    print('训练集中，总标签数为: ', len(total_labels))



anno_list = []
for i in range(1, 11):
    anno_list.append(f'/data/projects/iid_dataset/client{i}/val/anno.json')
num_labels = 80

for anno_file in anno_list:
    anno = json.load(open(anno_file, 'r'))
    total_labels = set()
    # 对于每一张图片的信息,统计其出现次数
    for info in anno:
        for label in info['labels']:
            total_labels.add(label)
    print('验证集中，总标签数为: ', len(total_labels))