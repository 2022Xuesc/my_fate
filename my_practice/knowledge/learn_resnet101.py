import torch.nn
import torchvision.models as torch_models

pretrained = True
num_labels = 1000

model = torch_models.resnet101(pretrained=pretrained, num_classes=num_labels)

torch.nn.Linear(in_features=2048, out_features=80)

print("Hello World")
