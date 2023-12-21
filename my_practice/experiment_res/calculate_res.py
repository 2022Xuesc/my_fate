import os
import csv
import statistics


def calculate_stats(float_list):
    minimum = min(float_list)
    maximum = max(float_list)
    mean = statistics.mean(float_list)
    variance = statistics.stdev(float_list)
    return minimum, maximum, mean


# paths = ['sync_fpsl_resnet','sync_fpsl_fixed_ratio_drop','sync_fpsl_bn_only_split','sync_fpsl_st']
# paths = ['gcn/base_fpsl', 'gcn/c_gcn', 'gcn/p_gcn_fedavg', 'gcn/p_gcn_fpsl', 'gcn/sal_gl_scene_2_fedavg',
#          'gcn/sal_gl_scene_6_fedavg','gcn/knn_4_fedavg','gcn/salgl_4_fedavg']
# paths = ['gcn/knn_4_fedavg','gcn/salgl_4_fedavg','IJCNN/resnet_agg_salgl','IJCNN/resnet_kmeans_lrp_0.1',
#          'IJCNN/resnet_salgl']

# IJCNN相关
# Todo: 计算mAP指标、OF1指标和CF1指标

# paths = ['gcn/base_fpsl', 'IJCNN/resnet_salgl', 'IJCNN/resnet_agg_salgl', 'IJCNN/resnet_kmeans_lrp_0.1', 'gcn/c_gcn',
#          'gcn/p_gcn_fedavg']

paths = ['IJCNN/nuswide_kmeans']
for path in paths:
    clients_path = [os.path.join(path, 'guest/10')]
    for i in range(1, 10):
        clients_path.append(os.path.join(path, f'host/{i}'))
    # 读取每个路径下的valid.csv文件，取出最大值，添加到列表中
    mAPs = []
    CF1s = []
    OF1s = []

    for i in range(len(clients_path)):
        with open(os.path.join(clients_path[i], 'valid.csv'), 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            mAP = 0
            CF1 = 0
            OF1 = 0
            for row in reader:
                mAP = max(mAP, float(row.get('map')))
                CF1 = max(CF1, float(row.get('CF1')))
                OF1 = max(OF1, float(row.get('OF1')))
            mAPs.append(mAP)
            CF1s.append(CF1 * 100)
            OF1s.append(OF1 * 100)
    # 计算列表中的最小值、最大值、均值和方差
    WmAP, BmAP, AmAP = calculate_stats(mAPs)
    WCF1, BCF1, ACF1 = calculate_stats(CF1s)
    WOF1, BOF1, AOF1 = calculate_stats(OF1s)
    print('————————————————————————————————————')
    print(path)
    print(f"mAP: AmAP, WmAP, BmAP = {AmAP:.1f}, {WmAP:.1f}, {BmAP:.1f}")
    print(f"CF1: ACF1, WCF1, BCF1 = {ACF1:.1f}, {WCF1:.1f}, {BCF1:.1f}")
    print(f"OF1: ACF1, WCF1, BCF1 = {AOF1:.1f}, {WOF1:.1f}, {BOF1:.1f}")
    print(
        f"{AmAP:.1f} & {WmAP:.1f} & {BmAP:.1f} & {ACF1:.1f} & {WCF1:.1f} & {BCF1:.1f} & {AOF1:.1f} & {WOF1:.1f}& {BOF1:.1f}")
    print('————————————————————————————————————')
