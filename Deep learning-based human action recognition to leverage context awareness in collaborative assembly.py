import torch
import torch.nn as nn
import torchvision.models as models

class ResNetLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=512, num_layers=1):
        super(ResNetLSTM, self).__init__()
        # Load pre-trained ResNet-34
        self.resnet = models.resnet34(pretrained=True)
        # Remove the final FC layer of ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Output: [batch, 512, 1, 1]
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Temporal consistency filter (simplified as a post-processing step, not part of the model graph)
        self.temporal_window = 2.5  # 2.5 seconds (PAGE 13)

    def forward(self, x):
        # x: [batch_size, 3, 16, 224, 224]
        batch_size, C, T, H, W = x.shape
        
        # Process each frame through ResNet
        x = x.permute(0, 2, 1, 3, 4)  # [batch_size, 16, 3, 224, 224]
        x = x.reshape(batch_size * T, C, H, W)  # [batch_size * 16, 3, 224, 224]
        x = self.resnet(x)  # [batch_size * 16, 512, 1, 1]
        x = x.view(batch_size, T, -1)  # [batch_size, 16, 512]
        
        # LSTM over the temporal dimension
        x, _ = self.lstm(x)  # [batch_size, 16, hidden_size]
        x = x[:, -1, :]  # Take the last time step: [batch_size, hidden_size]
        
        # Classification
        x = self.fc(x)  # [batch_size, num_classes]
        return x
