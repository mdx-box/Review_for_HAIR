import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel attention
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x))))
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial = torch.max(x, dim=1, keepdim=True)[0]
        spatial_att = torch.sigmoid(self.conv(torch.cat([avg_spatial, max_spatial], dim=1)))
        x = x * spatial_att
        return x


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, num_nodes=15):
        super(GraphConv, self).__init__()
        self.num_nodes = num_nodes
        # Static adjacency matrix (initialized with SGD)
        self.static_adj = nn.Parameter(torch.ones(num_nodes, num_nodes) * 0.1)
        self.static_weight = nn.Linear(in_features, out_features)
        # Dynamic adjacency matrix (learned per input)
        self.dynamic_weight = nn.Linear(in_features, out_features)

    def forward(self, x, features):
        # x: [batch_size, in_features, num_nodes] (e.g., from ResNet features)
        # Static GCN
        static_out = F.leaky_relu(torch.matmul(self.static_adj, x) @ self.static_weight.weight)
        # Dynamic GCN (simplified: adjacency learned from features)
        dynamic_adj = torch.softmax(torch.matmul(features, features.transpose(-1, -2)), dim=-1)
        dynamic_out = F.leaky_relu(torch.matmul(dynamic_adj, x) @ self.dynamic_weight.weight)
        return static_out + dynamic_out  # Joint training



from torchvision.models import resnet101

class AMGCN(nn.Module):
    def __init__(self, num_classes=15):
        super(AMGCN, self).__init__()
        # ResNet-101 backbone
        self.backbone = resnet101(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove final FC layer
        # Attention and multi-scale modules
        self.cbam = CBAM(2048)
        self.mls = MultiScaleFusion(2048)
        # Graph convolution
        self.gcn = GraphConv(2048, 512, num_nodes=num_classes)
        self.fc = nn.Linear(512 * num_classes, num_classes)

    def forward(self, x):
        # x: [batch_size, 3, 448, 448] or [batch_size, seq_len, 3, 448, 448]
        if x.dim() == 5:  # Sequence input
            batch_size, seq_len, c, h, w = x.shape
            x = x.view(batch_size * seq_len, c, h, w)
        
        # Backbone feature extraction
        features = self.backbone(x)  # [batch_size * seq_len, 2048, 14, 14]
        features = self.cbam(features)
        features = self.mls(features)
        
        # Global average pooling
        features = F.adaptive_avg_pool2d(features, (1, 1)).view(-1, 2048)  # [batch_size * seq_len, 2048]
        if x.dim() == 5:
            features = features.view(batch_size, seq_len, 2048).mean(dim=1)  # [batch_size, 2048]
        
        # Graph convolution (assuming 15 nodes for 15 behaviors)
        gcn_in = features.unsqueeze(-1).repeat(1, 1, 15)  # [batch_size, 2048, 15]
        gcn_out = self.gcn(gcn_in, features)  # [batch_size, 512, 15]
        out = self.fc(gcn_out.view(gcn_out.size(0), -1))  # [batch_size, 15]
        return out