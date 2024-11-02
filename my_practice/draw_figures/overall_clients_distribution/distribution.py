import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

rcParams['axes.unicode_minus'] = False
rcParams['font.family'] = 'SimHei'


plt.figure(figsize=(18,9))
consume = ['c' + str(i + 1) for i in range(10)]
# 注意：
samples = [[4500, 4856, 76469],
           [107, 30, 630],
           [27, 62, 949],
           [19, 16, 759],
           [226, 292, 500],
           [21, 152, 1051],
           [55, 81, 601],
           [10, 16, 658],
           [40, 52, 178],
           [6, 163, 286]
           ]
log_samples = np.log10(samples)
pdf = pd.DataFrame(data=log_samples, columns=['VOC 2007', 'VOC 2012', 'MS-COCO'], index=consume)

index = pdf.index
col = pdf.columns
width = 8
plt.bar([i * 30 + width * 0 for i in range(len(index))], pdf['VOC 2007'], width=width, label="VOC 2007")
plt.bar([i * 30 + width * 1 for i in range(len(index))], pdf['VOC 2012'], width=width, label="VOC 2012")
plt.bar([i * 30 + width * 2 for i in range(len(index))], pdf['MS-COCO'], width=width, label="MS-COCO")

for i, v in enumerate(log_samples):
    for j in range(3):
        plt.text(i * 30 + width * j, v[j] + 0.03, str(samples[i][j]), ha='center',
                 fontsize=10)

plt.xticks([x * 30 + width for x in range(10)], index, fontsize=16)
plt.xlim(-10, 300)
# plt.title(f'Distribution of three datasets among clients')
plt.ylabel('The size of the client dataset (log10)',fontsize=16)
plt.legend(loc=0)

plt.savefig(f'overall_fig.svg', dpi=600, format='svg')

