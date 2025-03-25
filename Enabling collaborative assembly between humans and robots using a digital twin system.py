import torch
import torch.nn as nn
import torch.nn.functional as F

# 图卷积层
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GraphConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.A = A  # 邻接矩阵 (V, V)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, C, T, V = x.size()
        # 图卷积：x = A * x * W
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

# 时空卷积块
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1):
        super(STGCNBlock, self).__init__()
        self.spatial_conv = GraphConv(in_channels, out_channels, A)
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = self.bn(x)
        return F.relu(x)

# EST-GCN 网络
class ESTGCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, num_joints=25, A=None):
        super(ESTGCN, self).__init__()
        self.A = A  # 邻接矩阵

        # ST-GCN 主干网络
        self.layer1 = STGCNBlock(in_channels, 64, A)
        self.layer2 = STGCNBlock(64, 64, A)
        self.layer3 = STGCNBlock(64, 128, A, stride=2)
        self.layer4 = STGCNBlock(128, 128, A)
        self.layer5 = STGCNBlock(128, 256, A, stride=2)
        self.layer6 = STGCNBlock(256, 256, A)

        # 全局平均池化
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头
        self.fc = nn.Linear(256, num_classes)

        # 不确定性估计分支（假设为简单的全连接层）
        self.uncertainty_fc = nn.Linear(256, num_classes)  # 输出每个类的不确定性

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        # 池化
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # 分类输出
        logits = self.fc(x)

        # 不确定性估计
        uncertainty = self.uncertainty_fc(x)
        uncertainty = torch.softmax(uncertainty, dim=1)  # 转换为概率分布

        return logits, uncertainty

