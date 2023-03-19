import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())
    t = torch.tensor([1, 2, 3]).type(torch.int8)
    t = t.to('cuda')
    print(t)
