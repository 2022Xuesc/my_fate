import csv
import os
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

# paths = ['IJCNN/nuswide_kmeans']
# paths = ['IJCNN/coco_fedavg']
# paths = ['IJCNN/salgl_no_agg', 'IJCNN/p_gcn_no_agg', 'IJCNN/c_gcn_no_agg']

paths = [
    # 'IJCNN/kmeans_agg',
    # 'AAAI/coco/add_gcn',
    # 'AAAI/coco/add_prob_gcn',
    # 'AAAI/coco/add_gap_gcn',
    # 'AAAI/coco/pruned_add_gcn',
    # 'AAAI/coco/connect_add_gcn',
    # 'AAAI/coco/fat_connect_prob_gcn',
    # 'AAAI/coco/gin',
    # 'AAAI/coco/add_residual_gcn',
    # 'AAAI/coco/add_standard_residual_gcn',
    # 'AAAI/coco/add_standard_residual_keep_static_gcn',
    # 'AAAI/coco/add_residual_keep_static_gcn',
    # 'AAAI/coco/fixed_connect_standard_gcn',
    # 'AAAI/coco/fixed_connect_prob_standard_gcn',
    # 'AAAI/coco/fixed_connect_prob_add_gcn',
    # 'AAAI/coco/connect_prob_standard_gcn',
    # 'AAAI/coco/fixed_prob_standard_gcn',

    # 'gcn/c_gcn',
    # 'gcn/p_gcn_fedavg'
    # 'AAAI/coco2017/fed_avg',
    # 'AAAI/coco2017/fed_avg_new',
    # 'AAAI/coco2017/flag',
    # 'AAAI/coco2017/fpsl',
    # 'AAAI/coco2017/fixed_connect_prob_standard_gcn',
    # 'AAAI/coco2017/add_gcn_origin'
    # 'AAAI/coco2017/connect_prob_standard_gcn',
    # 'AAAI/coco2017/fixed_prob_standard_gcn',
    # 'AAAI/coco2017/fixed_connect_standard_gcn',
    # 'AAAI/coco2017/fixed_connect_prob_gcn',
    # 'AAAI/coco2017/connect_prob_standard_gcn',
    # 'AAAI/coco2017/c_gcn',
    # 'AAAI/coco2017/p_gcn',
    # 'AAAI/coco/c_gcn',
    # 'AAAI/coco/p_gcn',
    # 'AAAI/coco/c_gcn_new',
    # 'AAAI/coco/p_gcn_new',
    # 'AAAI/coco/latest_ours',
    # 'AAAI/coco/add_gcn_origin',
    # 'AAAI/coco/add_gcn_large_lr'

    # 'IJCNN/voc/p_gcn',
    # 'IJCNN/voc/p_gcn_without_agg'

    'IJCNN/coco/fed_avg',
    'IJCNN/coco/fpsl',
    'IJCNN/coco/c_gcn',
    'IJCNN/coco/c_gcn_no_agg',
    'IJCNN/coco/p_gcn',
    'IJCNN/coco/p_gcn_no_agg',
    'IJCNN/coco/salgl',
    'IJCNN/coco/salgl_no_agg',
    'IJCNN/coco/kmeans',
    'IJCNN/coco/kmeans_no_agg'
]

paths = [
    'AAAI/coco2017/fed_avg',
    'AAAI/coco2017/flag',
    'AAAI/coco2017/fpsl',
    'AAAI/coco2017/c_gcn',
    'AAAI/coco2017/p_gcn',
    ]

# # coco2014消融实验
# 
# paths = ['AAAI/coco/fixed_connect_prob_standard_gcn',
#          'AAAI/coco/fixed_connect_prob_gcn',
#          'AAAI/coco/connect_prob_standard_gcn',
#          'AAAI/coco/fixed_prob_standard_gcn',
#          'AAAI/coco/fixed_connect_standard_gcn']
# 
# # coco2017消融实验
paths = ['AAAI/coco2017/fixed_connect_prob_standard_gcn',
         'AAAI/coco2017/fixed_connect_prob_gcn',
         'AAAI/coco2017/connect_prob_standard_gcn',
         'AAAI/coco2017/fixed_prob_standard_gcn',
         'AAAI/coco2017/fixed_connect_standard_gcn']


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
    
    print(f"CF1: WCF1, BCF1, ACF1 = {WCF1:.1f}, {BCF1:.1f}, {ACF1:.1f}")
    print(f"OF1: WOF1, BOF1, AOF1 = {WOF1:.1f}, {BOF1:.1f}, {AOF1:.1f}")
    print(f"mAP: WmAP, BmAP, AmAP = {WmAP:.1f}, {BmAP:.1f}, {AmAP:.1f}")
    # print(
    #     f"{AmAP:.1f} & {WmAP:.1f} & {BmAP:.1f} & {ACF1:.1f} & {WCF1:.1f} & {BCF1:.1f} & {AOF1:.1f} & {WOF1:.1f}& {BOF1:.1f}")
    print('————————————————————————————————————')
