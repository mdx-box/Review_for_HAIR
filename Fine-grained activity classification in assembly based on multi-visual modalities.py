import torch
import torch.nn as nn
from torchvision.models import vgg16

class TwoStageNetwork(nn.Module):
    def __init__(self, num_classes=15, rnn_hidden_size=128, dropout=0.5):
        super(TwoStageNetwork, self).__init__()
        # First Stage: VGG-16 for RGB and Skeleton
        vgg = vgg16(pretrained=True)
        self.rgb_extractor = nn.Sequential(*list(vgg.features)[:-1])  # Remove maxpool5
        self.skeleton_extractor = nn.Sequential(*list(vgg.features)[:-1])  # Same for skeleton
        self.pool = nn.AdaptiveMaxPool2d((7, 7))  # Max pooling to 7x7
        self.flatten = nn.Flatten(start_dim=2)  # Flatten spatial dims
        
        # Feature dimension after VGG-16 (512 channels * 7 * 7)
        self.feature_dim = 512 * 7 * 7
        
        # Second Stage: LSTM for each modality
        self.rgb_lstm = nn.LSTM(self.feature_dim, rnn_hidden_size, num_layers=1, batch_first=True)
        self.skeleton_lstm = nn.LSTM(self.feature_dim, rnn_hidden_size, num_layers=1, batch_first=True)
        
        # Fusion and Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)  # Fusion of 2 modalities
        
    def forward(self, rgb, skeleton):
        # rgb, skeleton: [batch_size, 20, 3 or C_s, 224, 224]
        batch_size, seq_len, C, H, W = rgb.shape
        
        # First Stage: Feature Extraction
        rgb = rgb.view(batch_size * seq_len, 3, H, W)  # [batch*20, 3, 224, 224]
        skeleton = skeleton.view(batch_size * seq_len, C, H, W)
        
        rgb_features = self.rgb_extractor(rgb)  # [batch*20, 512, 14, 14]
        skeleton_features = self.skeleton_extractor(skeleton)  # [batch*20, 512, 14, 14]
        
        rgb_features = self.pool(rgb_features)  # [batch*20, 512, 7, 7]
        skeleton_features = self.pool(skeleton_features)  # [batch*20, 512, 7, 7]
        
        rgb_features = self.flatten(rgb_features)  # [batch*20, 512*7*7]
        skeleton_features = self.flatten(skeleton_features)  # [batch*20, 512*7*7]
        
        rgb_features = rgb_features.view(batch_size, seq_len, -1)  # [batch, 20, 512*7*7]
        skeleton_features = skeleton_features.view(batch_size, seq_len, -1)
        
        # Second Stage: LSTM
        rgb_out, _ = self.rgb_lstm(rgb_features)  # [batch, 20, 128]
        skeleton_out, _ = self.skeleton_lstm(skeleton_features)  # [batch, 20, 128]
        
        # Fusion (after RNN)
        fused = torch.cat((rgb_out, skeleton_out), dim=-1)  # [batch, 20, 256]
        fused = self.dropout(fused)
        
        # Classifier (using last timestep)
        out = self.fc(fused[:, -1, :])  # [batch, 15]
        return out
