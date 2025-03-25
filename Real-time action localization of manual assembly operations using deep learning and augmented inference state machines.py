import torch
import torch.nn as nn

# 定义 C3D 网络
class C3D(nn.Module):
    def __init__(self, num_classes=7):
        super(C3D, self).__init__()
        # C3D 网络结构
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc6 = nn.Linear(512 * 1 * 4 * 4, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch, channels, frames, height, width] = [batch, 3, 16, 112, 112]
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 512 * 1 * 4 * 4)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)
        return x

# 定义 C3D-OpticalFlow 网络（双流结构）
class C3DOpticalFlow(nn.Module):
    def __init__(self, num_classes=7):
        super(C3DOpticalFlow, self).__init__()
        # RGB 流
        self.rgb_stream = C3D(num_classes=num_classes)
        # 光流流
        self.flow_stream = C3D(num_classes=num_classes)
        # 融合层
        self.fusion = nn.Linear(num_classes * 2, num_classes)

    def forward(self, rgb, flow):
        # rgb: [batch, 3, 16, 112, 112]
        # flow: [batch, 2, 16, 112, 112]
        rgb_out = self.rgb_stream(rgb)
        # 调整光流输入通道（2 通道光流转换为 3 通道以匹配 C3D 输入）
        flow = flow.repeat(1, 3 // 2, 1, 1, 1)[:, :3, :, :, :]  # 复制通道
        flow_out = self.flow_stream(flow)
        # 融合两个流的输出
        combined = torch.cat((rgb_out, flow_out), dim=1)
        out = self.fusion(combined)
        return out
