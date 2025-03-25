import torch
import torch.nn as nn

# 定义AlexNet网络
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):  # 默认1000类（ImageNet），HRC任务中可调整
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 卷积层1: 输入224x224x3，输出55x55x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            # 池化层1: 输出27x27x96
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 卷积层2: 输出27x27x256
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # 池化层2: 输出13x13x256
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 卷积层3: 输出13x13x384
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 卷积层4: 输出13x13x384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 卷积层5: 输出13x13x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 池化层3: 输出6x6x256
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # 全连接层1: 6x6x256 -> 4096
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # 全连接层2: 4096 -> 4096
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # 全连接层3: 4096 -> num_classes
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 计算网络参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 实例化网络
model = AlexNet(num_classes=1000)  # 假设1000类（ImageNet原始设计）
print(model)

# 计算参数量
total_params = count_parameters(model)
print(f"Total number of parameters: {total_params}")

# 模拟输入（批次大小=1，3通道，224x224图像）
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print(f"Output shape: {output.shape}")