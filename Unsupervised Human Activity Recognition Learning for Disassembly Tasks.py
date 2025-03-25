import torch
import torch.nn as nn

class SeqVAE(nn.Module):
    def __init__(self, latent_dim=5):
        super(SeqVAE, self).__init__()
        # Encoder: CNN + GRU
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [batch*20, 16, 64, 64]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [batch*20, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [batch*20, 64, 16, 16]
            nn.ReLU()
        )
        self.flatten = nn.Flatten(start_dim=1)  # [batch*20, 64*16*16]
        self.gru = nn.GRU(64*16*16, 128, num_layers=1, batch_first=True)
        
        # Latent space
        self.fc_mu = nn.Linear(128, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(128, latent_dim)  # Log variance
        
        # Decoder: GRU + CNN (dual branches)
        self.gru_decoder = nn.GRU(latent_dim, 128, num_layers=1, batch_first=True)
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [batch*20, 64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [batch*20, 32, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # [batch*20, 16, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # [batch*20, 1, 128, 128]
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x: [batch_size, 20, 1, 128, 128]
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)  # [batch*20, 1, 128, 128]
        
        # Encode
        cnn_out = self.cnn_encoder(x)  # [batch*20, 64, 16, 16]
        cnn_flat = self.flatten(cnn_out)  # [batch*20, 64*16*16]
        cnn_flat = cnn_flat.view(batch_size, seq_len, -1)  # [batch, 20, 64*16*16]
        gru_out, _ = self.gru(cnn_flat)  # [batch, 20, 128]
        
        # Latent space
        mu = self.fc_mu(gru_out)  # [batch, 20, 5]
        logvar = self.fc_logvar(gru_out)  # [batch, 20, 5]
        z = self.reparameterize(mu, logvar)  # [batch, 20, 5]
        
        # Decode (current and future)
        decoder_out, _ = self.gru_decoder(z)  # [batch, 20, 128]
        decoder_out = decoder_out.view(batch_size * seq_len, 128, 1, 1)  # [batch*20, 128, 1, 1]
        recon = self.cnn_decoder(decoder_out)  # [batch*20, 1, 128, 128]
        recon = recon.view(batch_size, seq_len, 1, H, W)  # [batch, 20, 1, 128, 128]
        
        return recon, mu, logvar, z  # Recon (current), latent stats, and z

