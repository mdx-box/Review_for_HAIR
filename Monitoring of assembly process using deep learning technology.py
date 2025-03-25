import torch
import torch.nn as nn

# 定义改进的3D CNN网络（带批归一化）
class AssemblyAction3DCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(AssemblyAction3DCNN, self).__init__()
        self.features = nn.Sequential(
            # Conv1: 输入 16x112x112x1
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1),  # 输出：16x112x112x64
            nn.BatchNorm3d(64),  # 批归一化
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Pool1: 输出 8x56x56x64

            # Conv2
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),  # 输出：8x56x56x128
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Pool2: 输出 4x28x28x128

            # Conv3
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),  # 输出：4x28x28x256
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Pool3: 输出 2x14x14x256

            # Conv4
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),  # 输出：2x14x14x512
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Pool4: 输出 1x7x7x512

            # Conv5
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),  # 输出：1x7x7x512
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=1, stride=1),  # Pool5: 输出 1x7x7x512（文献未明确池化核大小，假设为1）
        )
        self.flatten = nn.Flatten()  # 展平：1x7x7x512 = 25088
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),  # 全连接层1
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Dropout防止过拟合
            nn.Linear(4096, num_classes),  # 全连接层2，输出9类
            nn.Softmax(dim=1),  # Softmax分类
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# 计算网络参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 实例化网络
model = AssemblyAction3DCNN(num_classes=9)
print(model)

# 计算参数量
total_params = count_parameters(model)
print(f"Total number of parameters: {total_params}")

# 模拟输入（批次大小=1，1通道，16帧，112x112图像）
input_tensor = torch.randn(1, 1, 16, 112, 112)  # [batch, channels, depth, height, width]
output = model(input_tensor)
print(f"Output shape: {output.shape}")