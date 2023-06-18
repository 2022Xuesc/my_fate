import json

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


import matplotlib.pyplot as plt

# 示例数据
x = ['A', 'B', 'C', 'D']
y = [10, 20, 15, 25]

# 创建柱状图
plt.bar(x, y)

# 在每个柱上显示值
for i, v in enumerate(y):
    plt.text(i, v + 1, str(v), ha='center')

# 设置图形标题和轴标签
plt.title('Bar Chart with Values')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图形
plt.show()
