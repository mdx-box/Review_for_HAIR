import torch
import torch.nn as nn

class ActRgnCNN(nn.Module):
    def __init__(self, height, num_classes=7):
        super(ActRgnCNN, self).__init__()
        self.height = height  # H: 136 (distance) or 2312 (angle)
        self.width = 32  # W: number of frames

        # Block 1: 2 Conv + MaxPool
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # [1, H, W] -> [32, H, W]
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # [32, H, W] -> [32, H, W]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [32, H, W] -> [32, H/2, W/2]
        )

        # Block 2: 2 Conv + MaxPool
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [32, H/2, W/2] -> [64, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # [64, H/2, W/2] -> [64, H/2, W/2]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [64, H/2, W/2] -> [64, H/4, W/4]
        )

        # Block 3: Flatten + FC + Softmax
        self.flatten_size = 64 * (self.height // 4) * (self.width // 4)
        self.block3 = nn.Sequential(
            nn.Flatten(),  # [64, H/4, W/4] -> [64 * (H/4) * (W/4)]
            nn.Linear(self.flatten_size, num_classes),  # [64 * (H/4) * (W/4)] -> [num_classes]
            nn.Dropout(0.5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
