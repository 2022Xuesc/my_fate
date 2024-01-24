import torch
import torch.nn as nn

from federatedml.nn.backend.communication.model_utils import MultiSubModel, SingleSubModel

subnet_strategies = ['progressive', 'dense', 'mixed']
fullnet_strategies = ['baseline', 'partial', 'layerwise', 'svcca']


# Todo: 分段的ResNet101模型

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.GroupNorm(2, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            # nn.GroupNorm(2, out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            # 当输入维度和输出维度不匹配时，使用1*1卷积来进行维度的匹配
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
                # nn.GroupNorm(2, out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    Todo: 这里用的是GroupNorm而不是BatchNorm2d
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.GroupNorm(2, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.GroupNorm(2, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            # nn.GroupNorm(2, out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
                # nn.GroupNorm(2, out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, strategy, num_stages, num_classes):
        super().__init__()

        self.strategy = strategy
        self.num_stages = num_stages
        # Todo: 模型的输出类别
        self.num_classes = num_classes

        self.in_channels = 64

        # 第一个卷积层创建，和pytorch官方实现一致
        # Todo: 把BatchNorm换成了GroupNorm
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            # nn.GroupNorm(2, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        # add to a list, which is prepared for progressive learning
        if self.num_stages == 8:
            self.module_splits = []
            # Todo: 这里的划分方案是每一层对半划分
            self.module_splits.append(nn.Sequential(self.conv1, self.conv2_x[:num_block[0] // 2]))
            self.module_splits.append(self.conv2_x[num_block[0] // 2:])
            self.module_splits.append(self.conv3_x[:num_block[1] // 2])
            self.module_splits.append(self.conv3_x[num_block[1] // 2:])
            self.module_splits.append(self.conv4_x[:num_block[2] // 2])
            self.module_splits.append(self.conv4_x[num_block[2] // 2:])
            self.module_splits.append(self.conv5_x[:num_block[3] // 2])
            self.module_splits.append(self.conv5_x[num_block[3] // 2:])

            self.head_splits = []
            # 根据每一层的维度创建对应的任务头
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(64 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(64 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(128 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(128 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(256 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(256 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(512 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(self.avg_pool,
                                                  nn.Flatten(),
                                                  self.fc))

        if self.num_stages == 5:
            self.module_splits = []
            self.module_splits.append(nn.Sequential(self.conv1,
                                                    self.conv2_x[:num_block[0] // 2]))
            self.module_splits.append(nn.Sequential(self.conv2_x[num_block[0] // 2:],
                                                    self.conv3_x[:num_block[1] // 2]))
            self.module_splits.append(nn.Sequential(self.conv3_x[num_block[1] // 2:],
                                                    self.conv4_x[:num_block[2] // 2]))
            self.module_splits.append(nn.Sequential(self.conv4_x[num_block[2] // 2:],
                                                    self.conv5_x[:num_block[3] // 2]))
            self.module_splits.append(self.conv5_x[num_block[3] // 2:])

            self.head_splits = []
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(64 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(128 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(256 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(512 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(self.avg_pool,
                                                  nn.Flatten(),
                                                  self.fc))
        elif self.num_stages == 4:
            self.module_splits = []
            self.module_splits.append(nn.Sequential(self.conv1,
                                                    self.conv2_x))
            self.module_splits.append(self.conv3_x)
            self.module_splits.append(self.conv4_x)
            self.module_splits.append(self.conv5_x)

            self.head_splits = []
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(64 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(128 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(256 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(self.avg_pool,
                                                  nn.Flatten(),
                                                  self.fc))
        elif self.num_stages == 3:
            self.module_splits = []
            self.module_splits.append(nn.Sequential(self.conv1,
                                                    self.conv2_x))
            self.module_splits.append(nn.Sequential(self.conv3_x,
                                                    self.conv4_x))
            self.module_splits.append(self.conv5_x)

            self.head_splits = []
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(64 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(256 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(self.avg_pool,
                                                  nn.Flatten(),
                                                  self.fc))

        elif self.num_stages == 2:
            self.module_splits = []
            self.module_splits.append(nn.Sequential(self.conv1,
                                                    self.conv2_x,
                                                    self.conv3_x,
                                                    self.conv4_x))
            self.module_splits.append(nn.Sequential(self.conv5_x))

            self.head_splits = []
            self.head_splits.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Flatten(),
                                                  nn.Linear(256 * block.expansion, self.num_classes)))
            self.head_splits.append(nn.Sequential(self.avg_pool,
                                                  nn.Flatten(),
                                                  self.fc))

        self.ind = -1
        self.enc = None
        self.head = None

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = x
        for m in self.module_splits:
            output = m(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

    def dense_forward(self, x):
        results = []
        out = x
        for i in range(self.num_stages):
            out = self.module_splits[i](out)
            results.append(self.head_splits[i](out))
        return results

    def set_submodel(self, ind, prev_model_splits):
        self.ind = ind
        assert ind <= self.num_stages - 1
        modules = prev_model_splits
        modules.append(self.module_splits[ind])
        self.enc = nn.Sequential(*modules)
        self.head = self.head_splits[ind]

    def get_submodel(self, stage_id, last_model=None):
        prev_model_splits = []
        if last_model is not None:
            prev_model_splits = list(last_model.enc._modules.values())
        self.set_submodel(stage_id, prev_model_splits)
        return SingleSubModel(self.enc, self.head, self.strategy, self.ind)

    def gen_submodel(self):
        if self.strategy == 'dense':
            return MultiSubModel(self.enc, self.head_splits[:self.ind + 1], self.strategy, self.ind)
        else:
            return SingleSubModel(self.enc, self.head, self.strategy, self.ind)

    def return_stage_parameters(self):
        out = []
        for i in range(len(self.module_splits)):
            num = 0
            for p in self.module_splits[i].parameters():
                num += torch.numel(p)
            for p in self.head_splits[i].parameters():
                num += torch.numel(p)
            out.append(num)
        return out

    def return_num_parameters(self):
        total = 0
        for p in self.trainable_parameters():
            total += torch.numel(p)

        return total


def resnet101(num_stages, num_classes, strategy="progressive"):
    return ResNet(BottleNeck, [3, 4, 23, 3], strategy, num_stages, num_classes=num_classes)


def warm_up(whole_model, cur_model):
    stage_id = whole_model.ind
    if stage_id == 0:
        return
    cur_model.train()

    freezed_list = cur_model.enc[:stage_id]
    freezed_list.requires_grad_(False)
    freezed_list.eval()

    warmup_lr = 0.0001

    # Todo: 优化器只考虑需要warm up的参数，包括新添加的卷积层和全连接层
    warmup_optimizer = torch.optim.SGD(params=cur_model.latest_parameters(), lr=warmup_lr, momentum=0.9,
                                       weight_decay=1e-4)
    print(warmup_optimizer.param_groups[0]['lr'])
    # 开始训练，优化


# if __name__ == '__main__':
    # num_stages = 8
    # whole_model = resnet101(8, 80)
    # last_model = None
    # iterations = 40
    # iters_per_stage = 5
    # for it in range(iterations):
    #     if it % iters_per_stage == 0:
    #         stage_id = it // iters_per_stage
    #         last_model = whole_model.get_submodel(stage_id, last_model)
    #         warm_up(whole_model, last_model)
