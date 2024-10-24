import csv
import os

# IJCNN相关
# Todo: 计算mAP指标、OF1指标和CF1指标

files = os.listdir(".")
# files = ['fedavg', 'flag', 'fpsl', 'c_gcn', 'p_gcn', 'salgl', 'fixed_connect_add_standard_gcn']

clients_path = ['guest/10']
for i in range(1, 10):
    clients_path.append(f'host/{i}')

for path in files:
    if path.startswith("compare") or not os.path.isdir(path):
        continue
    maxClientVal = 0
    minClientVal = 100
    avgClientVal = 0
    dominantClientVal = 0
    # 遍历每个client_path
    for client_path in clients_path:
        maxVal = 0
        with open(os.path.join(f'{path}/{client_path}/valid.csv'), 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                maxVal = max(maxVal, float(row.get('mAP')))
        maxClientVal = max(maxVal, maxClientVal)
        minClientVal = min(maxVal, minClientVal)
        avgClientVal += maxVal
        if client_path == 'guest/10':
            dominantClientVal = maxVal
    avgClientVal /= 10
    print(
        f'path = {path},min = {round(minClientVal, 2)}, max = {round(maxClientVal, 2)},dominant={round(dominantClientVal, 2)},avg = {round(avgClientVal, 2)}')
