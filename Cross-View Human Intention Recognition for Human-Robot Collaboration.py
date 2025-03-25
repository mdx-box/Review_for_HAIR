import torch
import torch.nn as nn
import torchvision.models as models

class CrossViewAutoEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(CrossViewAutoEncoder, self).__init__()
        # Encoder: Pre-trained CNN (e.g., ResNet-18)
        self.encoder = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # [batch, 512, 1, 1]
        
        # Intention Semantic Inducer: CNN with same input/output dim
        self.inducer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        )
        
        # Decoder: Inverse CNN (simplified as transposed conv)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output: [batch, 3, 224, 224]
        )

    def forward(self, x):
        # x: [batch_size, 3, 8, 224, 224]
        batch_size, C, T, H, W = x.shape
        x = x.view(batch_size * T, C, H, W)  # [batch_size * 8, 3, 224, 224]
        
        # Encode
        latent = self.encoder(x)  # [batch_size * 8, 512, 1, 1]
        
        # Intention Semantic Inducer
        semantics = self.inducer(latent)  # [batch_size * 8, 512, 1, 1]
        
        # Decode to reconstruct another view
        recon = self.decoder(semantics)  # [batch_size * 8, 3, 224, 224]
        recon = recon.view(batch_size, T, 3, H, W)  # [batch_size, 8, 3, 224, 224]
        
        return recon, semantics.view(batch_size, T, 512)  # Return reconstruction and semantics

class IntentionRecognitionLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=10):
        super(IntentionRecognitionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, 8, 512] (fused semantics from multiple views)
        x, _ = self.lstm(x)  # [batch_size, 8, 256]
        x = x[:, -1, :]  # Last time step: [batch_size, 256]
        x = self.fc(x)  # [batch_size, num_classes]
        return x

