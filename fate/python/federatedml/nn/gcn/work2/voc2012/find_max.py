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

dir_name = 'stats'
files = os.listdir(dir_name)

for path in files:
    if path.endswith("aps.csv"):
        continue
    with open(os.path.join(f'{dir_name}/{path}'), 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        maxVal = 0
        for row in reader:
            maxVal = max(maxVal, float(row.get('mAP')))
        print(f"{path} : {round(maxVal,2)}")
