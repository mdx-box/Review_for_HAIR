import torch
import torch.nn as nn

class ConvGTN(nn.Module):
    def __init__(self, num_classes=12, feature_dim=126, seq_length=100, num_heads=6, num_layers=4, hidden_dim=512):
        super(ConvGTN, self).__init__()
        
        # 时间嵌入层的 Conv1D
        self.conv1d = nn.Conv1d(in_channels=feature_dim, out_channels=hidden_dim, kernel_size=5, padding=2)
        self.linear_embed = nn.Linear(hidden_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        # 双塔 Transformer
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*2),
            num_layers=num_layers
        )
        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*2),
            num_layers=num_layers
        )
        
        # 门控融合
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_activation = nn.Sigmoid()
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # 输入形状: [batch_size, features, sequence_length] -> [8, 126, 100]
        batch_size, features, seq_length = x.size()
        
        # Conv1D 处理时间维度
        x = x.permute(0, 2, 1)  # [batch_size, seq_length, features]
        x = x.permute(0, 2, 1)  # [batch_size, features, seq_length] -> [8, 126, 100]
        x = self.conv1d(x)      # [batch_size, hidden_dim, seq_length] -> [8, 512, 100]
        x = x.permute(0, 2, 1)  # [batch_size, seq_length, hidden_dim] -> [8, 100, 512]
        x = self.linear_embed(x)  # [8, 100, 512]
        x = self.leaky_relu(x)
        
        # 时间塔
        temporal_out = self.temporal_transformer(x)  # [batch_size, seq_length, hidden_dim]
        
        # 空间塔（需要调整维度）
        spatial_in = x.permute(1, 0, 2)  # [seq_length, batch_size, hidden_dim]
        spatial_out = self.spatial_transformer(spatial_in)  # [seq_length, batch_size, hidden_dim]
        spatial_out = spatial_out.permute(1, 0, 2)  # [batch_size, seq_length, hidden_dim]
        
        # 门控融合
        combined = torch.cat((temporal_out, spatial_out), dim=-1)  # [batch_size, seq_length, hidden_dim*2]
        gate_weights = self.gate(combined)  # [batch_size, seq_length, hidden_dim]
        gate_weights = self.gate_activation(gate_weights)
        fused = temporal_out * gate_weights + spatial_out * (1 - gate_weights)  # [batch_size, seq_length, hidden_dim]
        
        # 池化到序列级别
        fused = fused.mean(dim=1)  # [batch_size, hidden_dim]
        
        # 分类
        output = self.classifier(fused)  # [batch_size, num_classes]
        return output
