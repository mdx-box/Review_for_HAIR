import torch
import torch.nn as nn

class V2VPoseNet(nn.Module):
    def __init__(self, num_joints=15):
        super(V2VPoseNet, self).__init__()
        # 假设体视显微镜网格为 64x64x64
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),  # 1x64x64x64 -> 32x64x64x64
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x64x64x64 -> 64x32x32x32
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x32x32x32 -> 128x16x16x16
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # 128x16x16x16 -> 256x8x8x8
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),  # 256x8x8x8 -> 512x4x4x4
            nn.ReLU(inplace=True),
        )
        self.flatten_dim = 512 * 4 * 4 * 4  # 32,768
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_joints * 3),  # 15 joints x 3 coordinates (x, y, z)
        )

    def forward(self, x):
        # 假设输入为体视显微镜网格 [batch, 1, 64, 64, 64]
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, 15, 3)  # Reshape to [batch, 15, 3]
        return x
