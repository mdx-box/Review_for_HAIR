import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class SegmentationCNN(nn.Module):
    def __init__(self, num_classes=3):  # e.g., background, hand, object
        super(SegmentationCNN, self).__init__()
        vgg = vgg16(pretrained=True).features
        
        # Encoder (downsampling)
        self.encoder = nn.Sequential(*list(vgg.children())[:23])  # Up to pool4
        
        # Decoder (upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, 4, stride=2, padding=1)
        )
        
    def forward(self, x):
        # x: [B, 3, 224, 224]
        enc = self.encoder(x)  # [B, 512, 14, 14]
        out = self.decoder(enc)  # [B, num_classes, 224, 224]
        return out

# Example instantiation
seg_model = SegmentationCNN(num_classes=3)
x_seg = torch.randn(1, 3, 224, 224)
seg_output = seg_model(x_seg)  # Pixel-wise class probabilities

class StateEstimationCNN(nn.Module):
    def __init__(self, num_states=10):  # e.g., 10 states for bird house assembly
        super(StateEstimationCNN, self).__init__()
        vgg = vgg16(pretrained=True)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_states)
        )
        
    def forward(self, x):
        # x: [B, 3, 224, 224]
        x = self.features(x)  # [B, 512, 14, 14]
        x = self.avgpool(x)  # [B, 512, 7, 7]
        x = x.view(x.size(0), -1)  # [B, 512*7*7]
        x = self.classifier(x)  # [B, num_states]
        return F.softmax(x, dim=1)  # Probability distribution

class ActionRecognition(nn.Module):
    def __init__(self, num_states=10):
        super(ActionRecognition, self).__init__()
        self.state_model = StateEstimationCNN(num_states)
        self.transition_matrix = torch.ones(num_states, num_states) / num_states  # Placeholder P_H
        
    def bayesian_inference(self, curr_state, state_probs):
        # curr_state: int (current state index)
        # state_probs: [B, num_states] from CNN
        transition_probs = self.transition_matrix[curr_state]  # [num_states]
        posterior = state_probs * transition_probs  # [B, num_states]
        posterior = posterior / posterior.sum(dim=1, keepdim=True)  # Normalize
        return posterior.argmax(dim=1)  # Predicted state
    
    def forward(self, x, curr_state):
        state_probs = self.state_model(x)  # [B, num_states]
        pred_state = self.bayesian_inference(curr_state, state_probs)  # [B]
        return pred_state

