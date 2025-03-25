import torch
import torch.nn as nn

class CTR_GCN(nn.Module):
    def __init__(self, in_channels=3, num_joints=25, num_classes=3, num_frames=100):
        """
        CTR-GCN 网络
        - in_channels: 输入通道数 (x, y, z 坐标 = 3)
        - num_joints: 关节数 (NTU RGB+D 格式通常为 25)
        - num_classes: 输出类别数 (3: key, gear, shaft)
        - num_frames: 输入帧数 (假设为 100)
        """
        super(CTR_GCN, self).__init__()
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.num_frames = num_frames

        # 初始邻接矩阵 (假设为 NTU RGB+D 的骨架拓扑)
        self.A = torch.ones(num_joints, num_joints)  # 简化版，实际需定义骨架连接

        # 特征变换层
        self.feature_transform = nn.Conv2d(in_channels, 64, kernel_size=1)

        # 通道级拓扑建模 (学习 Q)
        self.topology_conv = nn.Conv2d(64, 64, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 可训练标量 α

        # GCN 层
        self.gcn = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 时空池化
        self.temporal_pool = nn.AdaptiveAvgPool2d((1, 64))  # 池化时间维度
        self.fc = nn.Linear(64, num_classes)  # 输出层

    def forward(self, x):
        """
        - x: 输入骨架序列，形状 (batch_size, in_channels, num_frames, num_joints)
        返回：动作分类结果，形状 (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # 特征变换
        x = self.feature_transform(x)  # (batch_size, 64, num_frames, num_joints)

        # 通道级拓扑建模
        Q = self.topology_conv(x)  # 动态拓扑 (batch_size, 64, num_frames, num_joints)
        R = self.A + self.alpha * Q.mean(dim=2, keepdim=True)  # 简化 R 计算

        # GCN 特征提取
        x = self.gcn(x)  # (batch_size, 64, num_frames, num_joints)

        # 时空池化
        x = self.temporal_pool(x)  # (batch_size, 64, 1, 1)
        x = x.view(batch_size, -1)  # (batch_size, 64)

        # 输出分类
        logits = self.fc(x)  # (batch_size, num_classes)
        return logits

