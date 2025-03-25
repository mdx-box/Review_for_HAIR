import torch
import torch.nn as nn
import torchvision.models as models

# 定义一个简化的 Darknet-53 骨干网络（仅部分层，完整 Darknet-53 有 53 层）
class Darknet53Backbone(nn.Module):
    def __init__(self):
        super(Darknet53Backbone, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # 简化为较少层，实际 Darknet-53 更深
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

# 定义 YOLOv3 检测头（简化版，仅一个尺度）
class YOLOv3Head(nn.Module):
    def __init__(self, num_classes=11, num_anchors=3):
        super(YOLOv3Head, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # 每个 anchor 预测 (x, y, w, h, confidence) + num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, num_anchors * (5 + num_classes), kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.conv(x)
        # 调整输出形状为 [batch, num_anchors * (5 + num_classes), height, width]
        return x

# 完整的 YOLOv3 模型
class YOLOv3(nn.Module):
    def __init__(self, num_classes=11):
        super(YOLOv3, self).__init__()
        self.backbone = Darknet53Backbone()
        self.head = YOLOv3Head(num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
