from torchvision import models as torch_models
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pretrained_alexnet = torch_models.alexnet(pretrained=True)

    conv1_name = 'features.0'
    fc1_name = 'classifier.1'
    conv1_array = ...
    fc_array = ...
    for module_name,module in pretrained_alexnet.named_modules():
        if module_name == conv1_name:
            conv1_array = module.weight.data.view(-1).numpy()
        if module_name == fc1_name:
            fc_array = module.weight.data.view(-1).numpy()

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.hist(x = fc_array,
             bins = 200,range=(-0.05,0.05))
    plt.xlabel('权重值')
    plt.ylabel('权重数量')
    plt.title('Fc1层的权重分布情况')
    # plt.show()

    plt.savefig('Fc1.svg',format='svg')
    plt.show()
