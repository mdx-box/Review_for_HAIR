import torch
import torch.nn as nn

class SpatialStream(nn.Module):
    def __init__(self):
        super(SpatialStream, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)  # 120x160x16
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 60x80x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 60x80x32
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)  # 30x40x32

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.avgpool(x)
        return x

class TemporalStream(nn.Module):
    def __init__(self, num_frames=10):
        super(TemporalStream, self).__init__()
        self.num_frames = num_frames
        self.conv1 = nn.Conv2d(2, 64, kernel_size=9, stride=1, padding=4)  # 120x160x64
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 60x80x64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)  # 60x80x64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # 60x80x64
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)  # 60x80x32
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)  # 30x40x32

    def forward(self, x):
        # x: [batch, num_frames * 2, height, width]
        batch_size = x.size(0)
        # 将帧数和通道维度合并，模拟2D卷积处理
        x = x.view(batch_size, 2, 120, 160)  # 简化处理，实际需3D卷积或逐帧处理
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.avgpool(x)
        return x


class TwoStreamCNN(nn.Module):
    def __init__(self, num_classes=7, num_frames=10):
        super(TwoStreamCNN, self).__init__()
        self.spatial_stream = SpatialStream()
        self.temporal_stream = TemporalStream(num_frames=num_frames)
        # 冻结卷积和池化层（迁移学习）
        for param in self.spatial_stream.parameters():
            param.requires_grad = False
        for param in self.temporal_stream.parameters():
            param.requires_grad = False
        # 分类器
        self.fc1 = nn.Linear(30 * 40 * 64, 512)  # 融合特征：30x40x64
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)  # 7个类别
        self.softmax = nn.Softmax(dim=1)

    def forward(self, spatial_input, temporal_input):
        # 空间流
        spatial_features = self.spatial_stream(spatial_input)  # [batch, 32, 30, 40]
        # 时间流
        temporal_features = self.temporal_stream(temporal_input)  # [batch, 32, 30, 40]
        # 特征融合
        fused_features = torch.cat((spatial_features, temporal_features), dim=1)  # [batch, 64, 30, 40]
        # 展平
        fused_features = fused_features.view(fused_features.size(0), -1)  # [batch, 30*40*64]
        # 分类
        x = self.relu(self.fc1(fused_features))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


