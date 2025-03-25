import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(SpatialGCN, self).__init__()
        self.A = A  # Adjacency matrix [13, 13]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # x: [B, C, T, V]
        A_hat = self.A + torch.eye(13, device=x.device)  # Add self-loops
        D = torch.diag_embed(A_hat.sum(dim=1)).inverse().sqrt()
        A_norm = D @ A_hat @ D
        out = A_norm @ x.permute(0, 2, 3, 1)  # [B, T, V, C]
        out = out.permute(0, 3, 1, 2)  # [B, C, T, V]
        out = self.conv(out)
        out = self.bn(out)
        return F.relu(out)

class TemporalGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalGCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 6, 1)
        self.branches = nn.ModuleList([
            nn.Conv2d(out_channels // 6, out_channels // 6, (1, 1)),
            nn.MaxPool2d((3, 1), padding=(1, 0)),
            *[nn.Conv2d(out_channels // 6, out_channels // 6, (3, 1), padding=(i, 0), dilation=i) for i in range(1, 5)]
        ])
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        outs = [branch(x) for branch in self.branches]
        out = torch.cat(outs, dim=1)
        out = self.conv2(out)
        out = self.bn(out)
        return F.relu(out)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(STGCNBlock, self).__init__()
        self.s_gcn = SpatialGCN(in_channels, out_channels, A)
        self.t_gcn = TemporalGCN(out_channels, out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        out = self.s_gcn(x)
        out = self.t_gcn(out)
        return out + residual

class STGCNPlusPlus(nn.Module):
    def __init__(self, num_classes=60, A=None):
        super(STGCNPlusPlus, self).__init__()
        self.A = A if A is not None else torch.ones(13, 13)  # Placeholder adjacency
        self.blocks = nn.ModuleList([
            STGCNBlock(3, 64, self.A), STGCNBlock(64, 64, self.A), STGCNBlock(64, 64, self.A),
            STGCNBlock(64, 128, self.A), STGCNBlock(128, 128, self.A), STGCNBlock(128, 128, self.A),
            STGCNBlock(128, 256, self.A), STGCNBlock(256, 256, self.A), STGCNBlock(256, 256, self.A)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x: [B, T, V, C] -> [B, C, T, V]
        x = x.permute(0, 3, 1, 2)  # [B, 3, 32, 13]
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)  # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]
        return self.fc(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(SpatialGCN, self).__init__()
        self.A = A  # Adjacency matrix [13, 13]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # x: [B, C, T, V]
        A_hat = self.A + torch.eye(13, device=x.device)  # Add self-loops
        D = torch.diag_embed(A_hat.sum(dim=1)).inverse().sqrt()
        A_norm = D @ A_hat @ D
        out = A_norm @ x.permute(0, 2, 3, 1)  # [B, T, V, C]
        out = out.permute(0, 3, 1, 2)  # [B, C, T, V]
        out = self.conv(out)
        out = self.bn(out)
        return F.relu(out)

class TemporalGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalGCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 6, 1)
        self.branches = nn.ModuleList([
            nn.Conv2d(out_channels // 6, out_channels // 6, (1, 1)),
            nn.MaxPool2d((3, 1), padding=(1, 0)),
            *[nn.Conv2d(out_channels // 6, out_channels // 6, (3, 1), padding=(i, 0), dilation=i) for i in range(1, 5)]
        ])
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        outs = [branch(x) for branch in self.branches]
        out = torch.cat(outs, dim=1)
        out = self.conv2(out)
        out = self.bn(out)
        return F.relu(out)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(STGCNBlock, self).__init__()
        self.s_gcn = SpatialGCN(in_channels, out_channels, A)
        self.t_gcn = TemporalGCN(out_channels, out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        out = self.s_gcn(x)
        out = self.t_gcn(out)
        return out + residual

class STGCNPlusPlus(nn.Module):
    def __init__(self, num_classes=60, A=None):
        super(STGCNPlusPlus, self).__init__()
        self.A = A if A is not None else torch.ones(13, 13)  # Placeholder adjacency
        self.blocks = nn.ModuleList([
            STGCNBlock(3, 64, self.A), STGCNBlock(64, 64, self.A), STGCNBlock(64, 64, self.A),
            STGCNBlock(64, 128, self.A), STGCNBlock(128, 128, self.A), STGCNBlock(128, 128, self.A),
            STGCNBlock(128, 256, self.A), STGCNBlock(256, 256, self.A), STGCNBlock(256, 256, self.A)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x: [B, T, V, C] -> [B, C, T, V]
        x = x.permute(0, 3, 1, 2)  # [B, 3, 32, 13]
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)  # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]
        return self.fc(x)

class DualStreamModel(nn.Module):
    def __init__(self, num_classes=60, A=None):
        super(DualStreamModel, self).__init__()
        self.skel_stream = STGCNPlusPlus(num_classes, A)
        self.vid_stream = ResNet3D(num_classes)
        self.weight_skel = nn.Parameter(torch.tensor(0.5))
        self.weight_vid = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x_skel, x_vid):
        skel_out = self.skel_stream(x_skel)  # [B, num_classes]
        vid_out = self.vid_stream(x_vid)    # [B, num_classes]
        fused_out = self.weight_skel * skel_out + self.weight_vid * vid_out
        return fused_out

