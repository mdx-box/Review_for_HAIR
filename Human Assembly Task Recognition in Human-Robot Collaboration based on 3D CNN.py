import torch
import torch.nn as nn

# 定义3D CNN网络
class AssemblyTask3DCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(AssemblyTask3DCNN, self).__init__()
        self.features = nn.Sequential(
            # 第一阶段：2层3D卷积 + 1层3D最大池化
            # 输入：64x64x100x3
            nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1),  # 输出：64x64x100x32
            nn.Tanh(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),  # 输出：64x64x100x32
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=3, stride=3),  # 输出：22x22x34x32 (向下取整)
            # 第二阶段：2层3D卷积 + 1层3D最大池化
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),  # 输出：22x22x34x64
            nn.Tanh(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),  # 输出：22x22x34x64
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=3, stride=3),  # 输出：8x8x12x64 (向下取整)
        )
        self.flatten = nn.Flatten()  # 展平：8x8x12x64 = 49152
        self.classifier = nn.Sequential(
            nn.Linear(49152, 512),  # 全连接层1
            nn.Tanh(),
            nn.Linear(512, num_classes),  # 全连接层2，输出7类
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
model = AssemblyTask3DCNN(num_classes=7)
print(model)

# 计算参数量
total_params = count_parameters(model)
print(f"Total number of parameters: {total_params}")

# 模拟输入（批次大小=1，3通道，100帧，64x64图像）
input_tensor = torch.randn(1, 3, 100, 64, 64)  # [batch, channels, depth, height, width]
output = model(input_tensor)
print(f"Output shape: {output.shape}")