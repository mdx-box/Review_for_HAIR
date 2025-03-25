import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj, x) + self.bias
        return x

class GCN(nn.Module):
    def __init__(self, num_classes, num_joints=14, num_frames=120):
        super(GCN, self).__init__()
        self.num_joints = num_joints
        self.num_frames = num_frames

        self.gc1 = GraphConvolution(3, 64)  # 3 input channels (x, y, z coordinates)
        self.gc2 = GraphConvolution(64, 128)
        self.gc3 = GraphConvolution(128, 256)

        self.fc = nn.Linear(256 * num_joints * num_frames, num_classes)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
