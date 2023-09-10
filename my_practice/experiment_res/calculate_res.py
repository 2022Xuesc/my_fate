import os
import csv
import statistics


def calculate_stats(float_list):
    minimum = min(float_list)
    maximum = max(float_list)
    mean = statistics.mean(float_list)
    variance = statistics.stdev(float_list)
    return minimum, maximum, mean, variance


# paths = ['sync_fpsl_resnet', 'sync_fpsl_agg_bn', 'sync_fpsl_fixed_ratio_drop',
#          'sync_fpsl_lamp', 'sync_fpsl_dep_global', 'sync_fpsl_st', 'sync_fpsl_st_dep']
paths = ['sync_fpsl_resnet','sync_fpsl_fixed_ratio_drop','sync_fpsl_bn_only_split','sync_fpsl_st']
for path in paths:
    clients_path = [os.path.join(path, 'guest/10')]
    for i in range(1, 10):
        clients_path.append(os.path.join(path, f'host/{i}'))
    # 读取每个路径下的valid.csv文件，取出最大值，添加到列表中
    mAPs = []
    for i in range(len(clients_path)):
        with open(os.path.join(clients_path[i], 'valid.csv'), 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            mAP = 0
            for row in reader:
                mAP = max(mAP, float(row.get('mAP')))
            mAPs.append(mAP)
    # 计算列表中的最小值、最大值、均值和方差
    minimum, maximum, mean, variance = calculate_stats(mAPs)
    print('————————————————————————————————————')
    print(path)
    print(f"Minimum: {minimum:.1f}")
    print(f"Maximum: {maximum:.1f}")
    print(f"Mean: {mean:.1f}")
    print(f"Variance: {variance:.1f}")
    print('————————————————————————————————————')
