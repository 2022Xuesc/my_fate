import csv
import os
import statistics


def calculate_stats(float_list):
    minimum = min(float_list)
    maximum = max(float_list)
    mean = statistics.mean(float_list)
    variance = statistics.stdev(float_list)
    return minimum, maximum, mean


# paths = ['agg_salgl', 'kmeans', 'salgl', 'c_gcn_with_agg', 'c_gcn_without_agg', 'p_gcn_with_agg', 'p_gcn_without_agg']
# all_paths = os.listdir()
# paths = []
# for path in all_paths:
#     if os.path.isdir(path):
#         paths.append(path)
paths = ['c_gcn_without_agg', 'p_gcn_without_agg', 'salgl']
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
    WmAP, BmAP, AmAP = calculate_stats(mAPs)
    print(path)
    print(f"mAP: AmAP, WmAP, BmAP = {AmAP:.1f}, {WmAP:.1f}, {BmAP:.1f}")
    
    print('————————————————————————————————————')

# 计算每个标签的平均精度
num_labels = 20
for path in paths:
    clients_path = [os.path.join(path, 'guest/10')]
    for i in range(1, 10):
        clients_path.append(os.path.join(path, f'host/{i}'))
    # 维护ap值之和
    ap_list = [0 for _ in range(num_labels)]
    # 维护ap值的数目
    ap_cnt = [0 for _ in range(num_labels)]
    for i in range(len(clients_path)):
        # 读取valid_aps.csv文件
        with open(os.path.join(clients_path[i], 'valid_aps.csv'), 'r') as csv_file:
            reader = csv.reader(csv_file)
            max_ap = [-1 for _ in range(num_labels)]  # 统计各个epoch的最大值
            for row in reader:
                for label_id in range(num_labels):
                    max_ap[label_id] = max(max_ap[label_id], float(row[label_id]))
            for label_id in range(num_labels):
                # 说明ap值有效
                if max_ap[label_id] != -1:
                    ap_list[label_id] += max_ap[label_id]
                    ap_cnt[label_id] += 1
    avg_ap_list = [round((ap_list[i] / ap_cnt[i]) * 100, 1) for i in range(num_labels)]
    print(f'{path}', avg_ap_list)
