import torch
import torch.nn as nn
from torchvision.models import resnet50

class InflatedResNet50(nn.Module):
    def __init__(self):
        super(InflatedResNet50, self).__init__()
        base = resnet50(pretrained=True)
        # Replace 2D conv/pool with 3D
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layers = nn.Sequential(*list(base.children())[4:-2])  # Res2-Res5 blocks
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        # x: [batch_size, 3, 15, 640, 576]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layers(x)
        x = self.avg_pool(x)
        return x.view(x.size(0), -1)  # [batch_size, 2048]

class STGCN(nn.Module):
    def __init__(self, in_channels=3, num_joints=25, out_channels=256):
        super(STGCN, self).__init__()
        self.num_joints = num_joints
        self.adj = nn.Parameter(torch.ones(num_joints, num_joints) * 0.1)  # Adjacency matrix
        self.conv1 = nn.Conv2d(in_channels, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: [batch_size, 3, 25, 50]
        batch_size, C, N, T = x.shape
        x = x.view(batch_size, C, N * T)  # Flatten spatial-temporal
        x = F.relu(self.bn1(self.conv1(x)))  # [batch_size, 64, N*T]
        # Graph convolution
        adj = torch.softmax(self.adj, dim=-1)
        x = torch.einsum('bct,nt->bcn', x, adj).view(batch_size, 64, N, T)
        x = self.conv2(x.mean(dim=-1))  # [batch_size, 256, 25]
        x = self.pool(x).view(batch_size, -1)  # [batch_size, 256]
        return x


class MultimodalFusion(nn.Module):
    def __init__(self, num_classes=5):
        super(MultimodalFusion, self).__init__()
        self.visual = InflatedResNet50()
        self.skeleton = STGCN(in_channels=3, num_joints=25, out_channels=256)
        self.attention1 = IntermediateAttention()
        self.attention2 = IntermediateAttention()
        self.fc = nn.Linear(2048 + 256, num_classes)

    def forward(self, rgb, skel):
        # rgb: [batch_size, 3, 15, 640, 576], skel: [batch_size, 3, 25, 50]
        vis = self.visual(rgb)  # [batch_size, 2048]
        skel = self.skeleton(skel)  # [batch_size, 256]
        vis1, skel1 = self.attention1(vis, skel)
        vis2, skel2 = self.attention2(vis1, skel1)
        fused = torch.cat([vis2, skel2], dim=1)  # [batch_size, 2304]
        out = self.fc(fused)  # [batch_size, num_classes]
        return out
