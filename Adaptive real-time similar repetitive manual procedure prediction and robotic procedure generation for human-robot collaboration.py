import torch
import torch.nn as nn
import torch.nn.functional as F

# 空间图卷积层
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(GraphConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(1, 1))
        self.kernel_size = kernel_size
        self.out_channels = out_channels

    def forward(self, x, A):
        # x: (N, C, T, V), A: (K, V, V)
        N, C, T, V = x.size()
        x = self.conv(x)  # (N, C', T, V)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, self.out_channels, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))  # 空间图卷积
        return x.contiguous()

# 时空图卷积块
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.0, residual=True):
        super(STGCNBlock, self).__init__()
        self.gcn = GraphConv(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding=(kernel_size[0]//2, 0))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout > 0 else nn.Identity()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, 1, (stride, 1))

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        x = self.dropout(x)
        x = self.relu(x + res)
        return x

# STGCNPP模型
class STGCNPP(nn.Module):
    def __init__(self, in_channels, num_classes, num_joints, num_frames):
        super(STGCNPP, self).__init__()
        self.data_bn = nn.BatchNorm2d(in_channels * num_joints)
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, (9, 3)),
            STGCNBlock(64, 64, (9, 3)),
            STGCNBlock(64, 64, (9, 3)),
            STGCNBlock(64, 128, (9, 3), stride=2),
            STGCNBlock(128, 128, (9, 3)),
            STGCNBlock(128, 128, (9, 3)),
            STGCNBlock(128, 256, (9, 3), stride=2),
            STGCNBlock(256, 256, (9, 3)),
            STGCNBlock(256, 256, (9, 3)),
        ])
        self.fc = nn.Linear(256, num_classes)
        self.num_frames = num_frames
        self.num_joints = num_joints

        # 假设的邻接矩阵A（需根据实际骨架数据定义）
        self.A = torch.ones(3, num_joints, num_joints)  # 简化假设，实际需定义图结构

    def forward(self, x):
        N, C, T, V = x.size()  # (batch_size, channels, frames, joints)
        x = x.view(N, C * V, T, 1)
        x = self.data_bn(x)
        x = x.view(N, C, T, V)

        for layer in self.layers:
            x = layer(x, self.A)

        x = F.avg_pool2d(x, (x.size(2), 1))  # Global average pooling over time
        x = x.view(N, -1)
        x = self.fc(x)
        return x
