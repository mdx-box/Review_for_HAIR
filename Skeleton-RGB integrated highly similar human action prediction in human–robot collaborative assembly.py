import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

class SkeletonRGBModel(nn.Module):
    def __init__(self, num_classes=8, skeleton_hidden_size=128, dropout=0.5):
        super(SkeletonRGBModel, self).__init__()
        # RGB Branch: R(2+1)D
        self.rgb_extractor = r2plus1d_18(pretrained=True)
        self.rgb_fc = nn.Linear(512, 256)  # Adjust R(2+1)D output to 256
        
        # Skeleton Branch: LSTM (assuming joint branch is temporal)
        self.skeleton_lstm = nn.LSTM(input_size=19*3, hidden_size=skeleton_hidden_size, 
                                   num_layers=1, batch_first=True)
        
        # Fusion and Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256 + skeleton_hidden_size, num_classes)
        
    def forward(self, rgb, skeleton):
        # rgb: [batch_size, 3, T, H, W], skeleton: [batch_size, T, 19, 3]
        batch_size = rgb.shape[0]
        
        # RGB Branch
        rgb_features = self.rgb_extractor(rgb)  # [batch_size, 512]
        rgb_features = self.rgb_fc(rgb_features)  # [batch_size, 256]
        
        # Skeleton Branch
        skeleton = skeleton.view(batch_size, skeleton.shape[1], -1)  # [batch_size, T, 19*3]
        skeleton_out, _ = self.skeleton_lstm(skeleton)  # [batch_size, T, 128]
        skeleton_features = skeleton_out[:, -1, :]  # [batch_size, 128]
        
        # Fusion
        fused = torch.cat((rgb_features, skeleton_features), dim=-1)  # [batch_size, 384]
        fused = self.dropout(fused)
        out = self.fc(fused)  # [batch_size, num_classes]
        return out

