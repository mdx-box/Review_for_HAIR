import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# 定义 TCA 模块
class TCAModule(nn.Module):
    def __init__(self, in_channels):
        super(TCAModule, self).__init__()
        self.avg_pool_h = nn.AdaptiveAvgPool2d((1, None))  # 沿高度（时间维度）池化
        self.avg_pool_w = nn.AdaptiveAvgPool2d((None, 1))  # 沿宽度（关节维度）池化
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x: [batch, channels, H, W]
        avg_h = self.avg_pool_h(x)  # 沿时间维度池化
        avg_w = self.avg_pool_w(x)  # 沿关节维度池化
        identity = x
        # 拼接池化结果和原始特征
        concat = torch.cat([avg_h, avg_w, identity], dim=1)
        out = self.conv(concat)
        return out

# 定义动作识别网络
class ActionRecognitionNetwork(nn.Module):
    def __init__(self, num_classes=5, in_channels=3):
        super(ActionRecognitionNetwork, self).__init__()
        # 使用 ResNet-18 作为骨干网络
        resnet = resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后的全连接层和池化层

        # 添加空间 dropout
        self.dropout = nn.Dropout2d(p=0.5)

        # TCA 模块
        self.tca = TCAModule(in_channels=512)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头：三个全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [batch, channels, H, W] = [batch, 3, 244, 244]
        features = self.backbone(x)  # [batch, 512, 8, 8]
        features = self.dropout(features)
        features = self.tca(features)  # TCA 模块
        features = self.global_pool(features)  # [batch, 512, 1, 1]
        out = self.classifier(features)  # [batch, num_classes]
        return out
