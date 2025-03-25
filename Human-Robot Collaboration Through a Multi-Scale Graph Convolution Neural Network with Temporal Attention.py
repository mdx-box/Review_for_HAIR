import torch
import torch.nn as nn
import torch.nn.functional as F

class SGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, k=8):
        super(SGCN, self).__init__()
        self.k = k
        self.A = A  # Adjacency matrix [18, 18]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # x: [B, C, T, N]
        B, C, T, N = x.shape
        A_hat = self.A + torch.eye(N, device=x.device)  # Add self-loops
        multi_scale_out = []
        
        for k in range(self.k + 1):
            A_k = torch.matrix_power(A_hat, k)  # Higher-order adjacency
            A_k = (A_k >= 1).float()  # Binarize
            D_k = torch.diag_embed(A_k.sum(dim=1)).inverse().sqrt()  # Degree matrix
            A_norm = D_k @ A_k @ D_k  # Normalized adjacency
            out = A_norm @ x.permute(0, 2, 3, 1)  # [B, T, N, C]
            multi_scale_out.append(out.permute(0, 3, 1, 2))  # [B, C, T, N]
        
        out = sum(multi_scale_out)  # Aggregate multi-scale features
        out = self.conv(out)
        out = self.bn(out)
        return F.relu(out)

class ATCN(nn.Module):
    def __init__(self, in_channels, out_channels, T, tau=9):
        super(ATCN, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(tau, 1), padding=(tau//2, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # x: [B, C, T, N]
        B, C, T, N = x.shape
        x_attn = x.permute(0, 3, 2, 1).reshape(B * N, T, C)  # [B*N, T, C]
        attn_out, _ = self.attn(x_attn, x_attn, x_attn)
        attn_out = attn_out.reshape(B, N, T, C).permute(0, 3, 2, 1)  # [B, C, T, N]
        out = self.conv(attn_out)
        out = self.bn(out)
        return F.relu(out)

class MSSTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, k=8, tau=9):
        super(MSSTBlock, self).__init__()
        self.sgcn = SGCN(in_channels, out_channels, A, k)
        self.atcn = ATCN(out_channels, out_channels, T=30, tau=tau)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        residual = self.residual(x)
        out = self.sgcn(x)
        out = self.atcn(out)
        out = self.dropout(out + residual)
        return out

class MSSTModel(nn.Module):
    def __init__(self, num_classes=8, A=None):
        super(MSSTModel, self).__init__()
        self.A = A if A is not None else torch.ones(18, 18)  # Placeholder adjacency
        self.blocks = nn.ModuleList([
            MSSTBlock(2, 64, self.A), MSSTBlock(64, 64, self.A), MSSTBlock(64, 64, self.A),
            MSSTBlock(64, 128, self.A), MSSTBlock(128, 128, self.A), MSSTBlock(128, 128, self.A),
            MSSTBlock(128, 256, self.A), MSSTBlock(256, 256, self.A), MSSTBlock(256, 256, self.A)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x: [B, T, N, C] -> [B, C, T, N]
        x = x.permute(0, 3, 1, 2)  # [B, 2, 30, 18]
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)  # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]
        return self.fc(x)
