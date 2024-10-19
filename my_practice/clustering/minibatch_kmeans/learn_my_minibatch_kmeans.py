import torch
import numpy as np

from federatedml.nn.backend.gcn.kmeans_torch import kmeans

data_size, dims, num_clusters = 1000, 2, 3
x = np.random.randn(data_size, dims) / 6
x = torch.from_numpy(x).to('cuda:0')

scene_cnts = [0] * num_clusters

batch_size = 32
num_steps = data_size // batch_size
centers = None
for i in range(num_steps):
    start = i * batch_size
    end = (i + 1) * batch_size
    cluster_ids_x, centers = kmeans(X=x[start:end], num_clusters=num_clusters,
                                    initial_state=centers,
                                    distance='euclidean', scene_cnts=scene_cnts)
