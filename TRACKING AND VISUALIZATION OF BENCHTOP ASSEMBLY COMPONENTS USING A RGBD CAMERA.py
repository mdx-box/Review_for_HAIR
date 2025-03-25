import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 U-Net 编码器块
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(conv)
        return conv, pool

# 定义 U-Net 解码器块
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)  # 跳跃连接
        x = self.conv(x)
        return x

# 定义简化的 FFB6D 网络
class SimplifiedFFB6D(nn.Module):
    def __init__(self, num_classes=2, pose_output=7):  # num_classes 为分割类别，pose_output 为 6D 位姿 + 置信度
        super(SimplifiedFFB6D, self).__init__()
        # 编码器（RGB 和深度分支共享）
        self.enc1 = EncoderBlock(4, 64)  # 输入 4 通道（RGB + 深度）
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # 解码器
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        # 分割头
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)

        # 位姿估计头
        self.pose_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, pose_output)  # 输出 6D 位姿（3 位置 + 4 四元数）+ 置信度
        )

    def forward(self, x):
        # 编码
        enc1, pool1 = self.enc1(x)
        enc2, pool2 = self.enc2(pool1)
        enc3, pool3 = self.enc3(pool2)
        enc4, pool4 = self.enc4(pool3)

        # 瓶颈
        bottleneck = self.bottleneck(pool4)

        # 解码
        dec4 = self.dec4(bottleneck, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        # 分割输出
        segmentation = self.segmentation_head(dec1)

        # 位姿估计输出
        pose = self.pose_head(bottleneck)

        return segmentation, pose

