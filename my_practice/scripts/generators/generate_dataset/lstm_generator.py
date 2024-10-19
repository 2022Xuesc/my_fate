# 从train_image2labels.json中读取数据，统计每个标签出现次数
from dataset_generator import *

phase = 'val'
image2labels = get_image2labels(phase)
# 1. 遍历该dict统计出每个标签的出现频率,得到标签的排序位次
label_num = 90
frequency = [0] * label_num
for (image_id, label_vec) in image2labels.items():
    for label_id in label_vec:
        frequency[label_id - 1] += 1
print(frequency)
#
# 2. 对image2labels的每个list，指定键的排序
for (image_id, label_vec) in image2labels.items():
    f_list = []
    for label_id in label_vec:
        f_list.append(frequency[label_id - 1])
    label_vec.sort(key=lambda k: frequency[k - 1], reverse=True)
# 将排序后的image2labels写到本地

json_obj = json.dumps(image2labels)
file_obj = open(f'{phase}_lstm_image2labels.json', 'w')
file_obj.write(json_obj)
file_obj.close()

