import json
import os

# total_weights = 0
# for i in range(1, 11):
#     anno_file = f'client{i}/train/anno.json'
#     file_obj = open(anno_file, 'r')
#     obj = json.load(file_obj)
#     num_per_label = [0] * 80
#     for info in obj:
#         labels = info['labels']
#         # 遍历每个label，将对应位置的数字加上1
#         for label in labels:
#             num_per_label[label] += 1
#     # num_per_label计算weight
#     weight = 0
#     alpha = 0.3
#     for num in num_per_label:
#         weight += num ** alpha
#     print(weight)
#     total_weights += weight
# print('===============================')
# print(total_weights)

# 单个epoch每个客户端的FLAG聚合权重
# weights = [769.0023070434984,
#            147.35534141240132,
#            191.57772081913726,
#            159.45368787603152,
#            152.59370917717644,
#            156.4159971145925,
#            157.65429678216867,
#            157.87776344170246,
#            103.70591541283105,
#            89.88833595376258]

# 正确的聚合权重
weight = [1246.2886617819984,
          238.81240611792236,
          310.4816970248992,
          258.4196711232792,
          247.302001391808,
          253.49661755204045,
          255.5034760769801,
          255.86563879289457,
          168.07167592986622,
          145.6781246292006]

# import matplotlib.pyplot as plt
#
# # 示例数据
# x = ['A', 'B', 'C', 'D']
# y = [10, 20, 15, 25]
#
# # 创建柱状图
# plt.bar(x, y)
#
# # 在每个柱上显示值
# for i, v in enumerate(y):
#     plt.text(i, v + 1, str(v), ha='center')
#
# # 设置图形标题和轴标签
# plt.title('Bar Chart with Values')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
#
# # 显示图形
# plt.show()
#
#
# import os
# import shutil
#
# # 源目录和目标目录
# source_dir = './'
# target_dir = '/data/projects/fate/my_practice/draw_figures/anno_json_dir'
#
# # 遍历client[1-10]子目录
# for i in range(1, 11):
#     # 构造源文件路径和目标文件路径
#     source_file = os.path.join(source_dir, f'client{i}', 'anno.json')
#     target_file = os.path.join(target_dir, f'anno_{i}.json')
#
#     # 拷贝文件
#     shutil.move(source_file, target_file)
#
# print('文件拷贝完成')

# import matplotlib.pyplot as plt
#
#
# client_nums = 10
# client_names = [f'client{i + 1}' for i in range(client_nums)]
# # 画权重分布图
# # 聚合权重分布
# plt.figure(figsize=(6,6))
# weight = [1246.2886617819984,
#           238.81240611792236,
#           310.4816970248992,
#           258.4196711232792,
#           247.302001391808,
#           253.49661755204045,
#           255.5034760769801,
#           255.86563879289457,
#           168.07167592986622,
#           145.6781246292006]
# weight_sum = sum(weight)
# plt.bar(client_names, weight)
# prob = [w / weight_sum for w in weight]
# for i, v in enumerate(weight):
#     plt.text(i, v + 5, str(round(prob[i],2)),ha='center')


from torch.optim.lr_scheduler import *
import torchvision.models as models
from torch.optim import *

model = models.resnet50(pretrained=False)
# 设立设定了初始的学习率
optimizer = Adam(model.parameters(),weight_decay=1e-4)

# 定义指数衰减学习率调度程序
# exp_scheduler = ExponentialLR(optimizer,gamma=0.9)

# one_cycle_lr会根据总步数自动确定初始学习率，以及从初始学习率增加到最大学习率再到降低到最终学习率的调度方式
one_cycle_scheduler = OneCycleLR(optimizer, max_lr=0.1,steps_per_epoch=10,epochs=10)
num_epochs = 50
# 训练过程中循环迭代
for epoch in range(10):
    # 执行训练和更新参数的操作

    # 在每个epoch开始时以指数方式更新学习率
    # exp_scheduler.step(epoch=epoch)

    ...
    # Todo: 注意，实际的遍历步数不能超过预先定义好的总步数=epoch*steps_per_epoch
    for batch in range(4):
        ...

        print(f"Epoch {epoch + 1}, Batch {batch + 1}, Learning rate: {one_cycle_scheduler.get_last_lr()}")
        # 每个batch更新一次学习率
        one_cycle_scheduler.step()
        # 输出当前epoch和学习率
        # print(f"Epoch {epoch + 1}, Batch {batch + 1}, Learning rate: {one_cycle_scheduler.get_last_lr()}")
