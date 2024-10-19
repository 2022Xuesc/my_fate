import numpy as np
from torchvision import models, transforms
import cv2
from PIL import Image
from torch.nn import functional as F

resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
densenet121 = models.densenet121(pretrained=True)

resnet18.eval()
resnet50.eval()
densenet121.eval()

image_transform = transforms.Compose([
    # 将输入图片resize成统一尺寸
    transforms.Resize([224, 224]),
    # 将PIL Image或numpy.ndarray转换为tensor，并除255归一化到[0,1]之间
    transforms.ToTensor(),
    # 标准化处理-->转换为标准正太分布，使模型更容易收敛
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

feature_data = []


# 定义钩子函数，注册在卷积层，获取卷积层的输出
def feature_hook(model, input, output):
    feature_data.append(output.data.numpy())


# Todo: layer4是一个大的子模块，其包括了最后一个卷计层，也就是说
#  layer4的输出就是最后一个卷积层的输出
resnet18._modules.get('layer4').register_forward_hook(feature_hook)
resnet50._modules.get('layer4').register_forward_hook(feature_hook)
densenet121._modules.get('features').register_forward_hook(feature_hook)

# 获取fc层的权重，方便进行feature map的加权聚合
fc_weights_resnet18 = resnet18._modules.get('fc').weight.data.numpy()
fc_weights_resnet50 = resnet50._modules.get('fc').weight.data.numpy()
fc_weights_densenet121 = densenet121._modules.get('classifier').weight.data.numpy()

# 获取预测类别id
img = image_transform(Image.open("imgs/dog.jpeg")).unsqueeze(0)
out_resnet18 = resnet18(img)
out_resnet50 = resnet50(img)
out_densenet121 = densenet121(img)

predict_class_id_resnet18 = np.argmax(F.softmax(out_resnet18, dim=1).data.numpy())
predict_class_id_resnet50 = np.argmax(F.softmax(out_resnet50, dim=1).data.numpy())
predict_class_id_densenet121 = np.argmax(F.softmax(out_densenet121, dim=1).data.numpy())


# 获取CAM
def makeCAM(feature, weights, classes_id):
    print(feature.shape, weights.shape, classes_id)
    bz, nc, h, w = feature.shape
    cam = weights[classes_id].dot(feature.reshape(nc, h * w))
    cam = cam.reshape(h, w)
    # 归一化到[0,1]之间
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    # 转换为0-255的灰度图
    cam_gray = np.uint8(255 * cam)
    # 上采样到与网络输入图像的尺寸一致，并返回
    return cv2.resize(cam_gray, (224, 224))


cam_gray_resnet18 = makeCAM(feature_data[0], fc_weights_resnet18, predict_class_id_resnet18)
cam_gray_resnet50 = makeCAM(feature_data[1], fc_weights_resnet50, predict_class_id_resnet50)
cam_gray_densenet121 = makeCAM(feature_data[2], fc_weights_densenet121, predict_class_id_densenet121)

# 叠加CAM和原图，并保存图片
src_image = cv2.imread("imgs/dog.jpeg")
h, w, _ = src_image.shape

cam_color_resnet18 = cv2.applyColorMap(cv2.resize(cam_gray_resnet18, (w, h)), cv2.COLORMAP_HSV)
cam_color_resnet50 = cv2.applyColorMap(cv2.resize(cam_gray_resnet50,(w,h)),cv2.COLORMAP_HSV)
cam_color_densenet121 = cv2.applyColorMap(cv2.resize(cam_gray_densenet121,(w,h)),cv2.COLORMAP_HSV)

# 合并原图和cam
cam_resnet18 = src_image * 0.5 + cam_color_resnet18 * 0.5
cam_resnet50 = src_image * 0.5 + cam_color_resnet50 * 0.5
cam_densenet121 = src_image * 0.5 + cam_color_densenet121 * 0.5

cam_hstack = np.hstack((src_image, cam_resnet18, cam_resnet50, cam_densenet121))
cv2.imwrite("imgs/dog_hstack.jpg", cam_hstack)
# 可视化
Image.open("imgs/dog_hstack.jpg").show()