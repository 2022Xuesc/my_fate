import csv

import json
import pandas as pd
import os

category_path = '../../../my_practice/dataset/voc_expanded/category.json'
json_file = open(category_path, 'r')
categories = json.load(json_file)
categories = list(categories.keys())


# noinspection PyTypeChecker
def fix_aps(base_path, col0_name, mAP_file_name, aps_file_name):
    headers = [col0_name, *categories, "mAP"]

    # 从avgloss中读取mAP数据
    loss_data = pd.read_csv(os.path.join(base_path, mAP_file_name))
    mAP = loss_data['mAP'].values
    ap_file_path = os.path.join(base_path, aps_file_name)

    # 需要写入的csv文件
    target_file_path = os.path.join(base_path, 'fixed_aps.csv')
    target_file = open(target_file_path, 'w', newline='')
    writer = csv.writer(target_file)
    writer.writerow(headers)

    col0_index = 0
    with open(ap_file_path, 'r') as ap_file:
        for line in ap_file:
            if line == '\n':
                continue
            row = [col0_index, *[float(ap) for ap in line.split(',')], mAP[col0_index]]
            writer.writerow(row)
            col0_index += 1
    print("Done")


# experiment_path = 'voc/voc_fpsl_200_onecycle'
#
# clients_path = [os.path.join(experiment_path, 'guest/10')]
# for i in range(1, 10):
#     clients_path.append(os.path.join(experiment_path, f'host/{i}'))
# for client_path in clients_path:
#     fix_aps(client_path, "epoch", "valid.csv", "val_aps.csv")
#
# arbiter_path = os.path.join(experiment_path, 'arbiter/999')
# fix_aps(arbiter_path, "agg_iter", "avgloss.csv", "agg_ap.csv")

path = "/home/klaus125/research/fate/state/standalone_res/plateau_stats"
fix_aps(path, "epoch", "valid.csv", "val_aps.csv")
