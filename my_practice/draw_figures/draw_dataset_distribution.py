import os
import matplotlib.pyplot as plt
import json
import numpy as np

def get_sample_size(data_dir):
    anno_path = os.path.join(data_dir, 'anno.json')
    anno = json.load(open(anno_path, 'r'))
    return len(anno)

client_nums = 10

# server_path = '/data/projects/clustered_dataset'

server_path = 'anno_json_dir'

save_dir = 'clusters_distribution'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

client_names = [f'c{i + 1}' for i in range(client_nums)]

total_labels = []

# Todo: 记录每个客户端的样本数量，画出直方图
samples = []

plt.figure(figsize=(12, 12))

# 画直方图
for i in range(client_nums):
    client_id = i + 1
    client_train_path = os.path.join(server_path, f'client{client_id}/train')
    client_valid_path = os.path.join(server_path, f'client{client_id}/val')

    # Todo: 作出分布直方图
    # draw_hist(save_dir, [client_train_path, client_valid_path])

    # labels = get_labels_cnts(client_train_path)

    # total_labels.append(get_labels_feature(labels))

    # 只选取训练集大小
    samples.append(get_sample_size(client_train_path))

# div_frame = calc_kl_divergence(client_names=client_names, label_tensors=torch.Tensor(total_labels))
# draw_heatmap(save_dir, div_frame)

plt.figure(figsize=(12, 12))
# Todo: 作出关于samples的柱状图
log_samples = np.log10(samples)

plt.bar(client_names, log_samples)

for i, v in enumerate(log_samples):
    plt.text(i, v + 0.05, str(samples[i]), ha='center')

plt.title('Distribution of MS-COCO datasets among clients')
plt.ylabel('The size of the client dataset (log10)')

plt.savefig(f'{save_dir}/dataset_distribution.svg', dpi=600, format='svg')