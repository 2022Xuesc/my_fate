import csv
import os
import statistics


def calculate_stats(float_list):
    minimum = min(float_list)
    maximum = max(float_list)
    mean = statistics.mean(float_list)
    variance = statistics.stdev(float_list)
    return minimum, maximum, mean


# IJCNN相关
# Todo: 计算mAP指标、OF1指标和CF1指标

files = os.listdir(".")

clients_path = ['guest/10']
for i in range(1, 10):
    clients_path.append(f'host/{i}')

for path in files:
    if path.startswith("compare") or not os.path.isdir(path):
        continue
    maxClientVal = 0
    minClientVal = 100
    avgClientVal = 0
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
    avgClientVal /= 10
    print(f'path = {path},min = {round(minClientVal,2)}, max = {round(maxClientVal,2)},  avg = {round(avgClientVal,2)}')
