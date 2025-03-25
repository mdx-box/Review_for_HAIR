import torch
import torch.nn as nn

class STGCN(nn.Module):
    def __init__(self, in_channels=3, num_joints=25, num_classes=5, graph_args=None):
        super(STGCN, self).__init__()
        # Graph structure (simplified adjacency matrix, typically predefined)
        self.graph = graph_args if graph_args else torch.ones(num_joints, num_joints)
        
        # Spatial-temporal GCN layers
        self.st_gcn_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 64, kernel_size=(3, 1), padding=(1, 0)),  # Temporal conv
            nn.Conv2d(64, 64, kernel_size=(1, 1)),                          # Spatial conv
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(128, 128, kernel_size=(1, 1)),
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x: [batch_size, channels=3, frames=150, joints=25]
        for layer in self.st_gcn_layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.pool(x)  # [batch_size, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [batch_size, 128]
        out = self.fc(x)  # [batch_size, num_classes]
        return out

from torchvision.models import resnet50  # Placeholder backbone
from torch.nn import functional as F

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_max = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        # Channel attention
        avg_out = self.fc2(F.relu(self.fc1(self.channel_avg(x).squeeze())))
        max_out = self.fc2(F.relu(self.fc1(self.channel_max(x).squeeze())))
        channel_att = torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        x = x * channel_att
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial = torch.max(x, dim=1, keepdim=True)[0]
        spatial_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1)))
        x = x * spatial_att
        return x

class ImprovedYOLOX(nn.Module):
    def __init__(self, num_classes=5):
        super(ImprovedYOLOX, self).__init__()
        self.backbone = resnet50(pretrained=True)  # Simplified as CSPDarknet not directly available
        self.cbam = CBAM(channels=2048)
        self.head = nn.Conv2d(2048, num_classes * 5, kernel_size=1)  # 5: (x, y, w, h, conf)
        
    def forward(self, x):
        # x: [batch_size, 3, H, W]
        x = self.backbone(x)  # [batch_size, 2048, H/32, W/32]
        x = self.cbam(x)
        out = self.head(x)  # [batch_size, num_classes * 5, H/32, W/32]
        return out

class HRCAssemblyIntention(nn.Module):
    def __init__(self, num_action_classes=5, num_part_classes=5):
        super(HRCAssemblyIntention, self).__init__()
        self.st_gcn = STGCN(in_channels=3, num_joints=25, num_classes=num_action_classes)
        self.yolox = ImprovedYOLOX(num_part_classes)
        
    def forward(self, skeleton, image):
        # skeleton: [batch_size, 3, 150, 25]
        # image: [batch_size, 3, H, W]
        action_pred = self.st_gcn(skeleton)  # [batch_size, 5]
        part_pred = self.yolox(image)        # [batch_size, 5*5, H/32, W/32]
        return action_pred, part_pred

# Rule-based reasoning (simplified as post-processing)
def rule_based_reasoning(action_pred, part_pred, part_threshold=0.5):
    action = torch.argmax(action_pred, dim=1)  # Predicted action
    parts = torch.sigmoid(part_pred[:, 4::5]) > part_threshold  # Confidence scores
    # Logic based on Table 11 and Fig. 12
    intention = []
    for a, p in zip(action, parts):
        if a == 0 and p[0] and p[1]:  # A1 & P1 & P2
            intention.append("Key assembly, next: gear")
        # Add more rules as per Fig. 12
    return intention

