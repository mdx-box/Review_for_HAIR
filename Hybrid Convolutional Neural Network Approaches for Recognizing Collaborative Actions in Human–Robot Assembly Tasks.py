import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(STGCNLayer, self).__init__()
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, (1, 1))  # Spatial conv
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, (kernel_size, 1))  # Temporal conv
        self.bn = nn.BatchNorm2d(out_channels)
        self.adj = nn.Parameter(torch.ones(24, 24) * 0.1)  # Adjacency matrix for 24 joints

    def forward(self, x):
        # x: [batch_size, 3, 50, 24]
        batch_size, C, T, V = x.shape
        # Spatial convolution with graph structure
        adj = torch.softmax(self.adj, dim=-1)
        x = torch.einsum('bcnv,vm->bcnm', x, adj)  # Apply adjacency
        x = self.spatial_conv(x)  # [batch_size, out_channels, T, V]
        x = F.relu(self.bn(x))
        # Temporal convolution
        x = self.temporal_conv(x)  # [batch_size, out_channels, T', V]
        return x

class STGCN(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, num_layers=6):
        super(STGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.residuals = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            self.layers.append(STGCNLayer(in_c, hidden_channels))
            self.residuals.append(nn.Conv2d(in_c, hidden_channels, 1) if in_c != hidden_channels else nn.Identity())

    def forward(self, x):
        for layer, res in zip(self.layers, self.residuals):
            residual = res(x)
            x = layer(x) + residual  # Residual connection
        return x  # [batch_size, 64, T', 24]

class OneDCNN(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(OneDCNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: [batch_size, 64, T']
        x = F.relu(self.bn(self.conv1d(x)))
        return x  # [batch_size, 64, T']


class HybridCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(HybridCNN, self).__init__()
        self.bn_input = nn.BatchNorm2d(3)  # Input normalization
        self.stgcn = STGCN(in_channels=3, hidden_channels=64, num_layers=6)
        self.dropout = nn.Dropout(0.5)
        self.onedcnn = OneDCNN(in_channels=64, out_channels=64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(64 * 24, num_classes)  # Fully connected layer

    def forward(self, x):
        # x: [batch_size, 3, 50, 24]
        x = self.bn_input(x)  # Normalize input
        x = self.stgcn(x)  # [batch_size, 64, T', 24]
        x = self.dropout(x)  # Apply dropout
        x = x.mean(dim=3)  # [batch_size, 64, T'] (average over joints)
        x = self.onedcnn(x)  # [batch_size, 64, T']
        x = x.unsqueeze(-1)  # [batch_size, 64, T', 1]
        x = self.pool(x)  # [batch_size, 64, 1, 1]
        x = x.view(x.size(0), -1)  # [batch_size, 64]
        x = self.fc(x)  # [batch_size, num_classes]
        return x

