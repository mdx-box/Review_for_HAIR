import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import swin3d_b, Swin3D_B_Weights

class MultimodalSwin3D(nn.Module):
    def __init__(self, num_classes=61):
        super(MultimodalSwin3D, self).__init__()
        
        # RGB Swin3D-B 编码器
        self.rgb_encoder = swin3d_b(weights=Swin3D_B_Weights.DEFAULT)
        self.rgb_fc = nn.Linear(1024, num_classes)  # Swin3D-B stage 4 输出 1024 维
        
        # Depth Swin3D-B 编码器（调整输入通道为 1）
        self.depth_encoder = swin3d_b(weights=Swin3D_B_Weights.DEFAULT)
        self.depth_encoder.patch_embed.proj = nn.Conv3d(1, 96, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.depth_fc = nn.Linear(1024, num_classes)
        
    def forward(self, rgb, depth):
        # RGB 前向传播 [batch_size, 3, 16, 224, 224]
        rgb_features = self.rgb_encoder(rgb)  # [batch_size, 1024]
        rgb_logits = self.rgb_fc(rgb_features)  # [batch_size, num_classes]
        
        # Depth 前向传播 [batch_size, 1, 16, 224, 224]
        depth_features = self.depth_encoder(depth)  # [batch_size, 1024]
        depth_logits = self.depth_fc(depth_features)  # [batch_size, num_classes]
        
        # 晚期融合：平均概率
        rgb_probs = F.softmax(rgb_logits, dim=-1)
        depth_probs = F.softmax(depth_logits, dim=-1)
        fused_probs = (rgb_probs + depth_probs) / 2  # [batch_size, num_classes]
        
        return fused_probs

class FocalLoss(nn.Module):
    def __init__(self, gamma_init=2.0, gamma_final=0.1, total_epochs=20):
        super(FocalLoss, self).__init__()
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.total_epochs = total_epochs
        
    def forward(self, inputs, targets, epoch):
        # 计算当前 gamma（指数衰减）
        gamma = self.gamma_init * (self.gamma_final / self.gamma_init) ** (epoch / self.total_epochs)
        
        # 计算 Focal Loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

