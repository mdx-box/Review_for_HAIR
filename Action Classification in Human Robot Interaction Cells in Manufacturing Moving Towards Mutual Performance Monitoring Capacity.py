import torch
import torch.nn as nn
import torchvision.models as models

class OpticalFlowMobileNetV2SSD(nn.Module):
    def __init__(self, num_classes=14, input_channels=2):
        super(OpticalFlowMobileNetV2SSD, self).__init__()
        
        # 加载预训练的 MobileNetV2，并修改第一层以适应光流输入（2 通道）
        self.mobilenet = models.mobilenet_v2(pretrained=False)
        self.mobilenet.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # MobileNetV2 的特征提取部分
        self.features = self.mobilenet.features
        
        # SSD 分类头：替换 MobileNetV2 的分类器，添加适合动作分类的层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(1280, 512),  # MobileNetV2 最后一层的输出通道数为 1280
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)  # 提取特征
        x = self.classifier(x)  # 分类
        return x

