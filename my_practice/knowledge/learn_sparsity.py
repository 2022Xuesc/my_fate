import federatedml.nn.backend.distiller.utils as utils
import torch
import csv

if __name__ == '__main__':
    # t = torch.tensor([[1., 2., 0.], [0., 3., 0.], [1., 1., 0.], [0., 0, 1]])
    # f = open('sparsity.csv', 'w', encoding='utf-8')
    # csv_writer = csv.writer(f)
    # csv_writer.writerow(['epoch', 'conv_sparsity', 'linear_sparsity', 'total_sparsity'])
    # csv_writer.writerow([1, 0.2, 0.2, 0.2])
    for i in range(0,10,2):
        print(i)
