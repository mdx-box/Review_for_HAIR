import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

# 定义基于 ViT 的工具识别网络
class ToolRecognitionViT(nn.Module):
    def __init__(self, num_classes=12, pretrained=True):
        super(ToolRecognitionViT, self).__init__()
        
        # 加载预训练的 ViT 模型 (ViT-B/16)
        if pretrained:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        else:
            config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
            self.vit = ViTModel(config)
        
        # 修改分类头
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)  # hidden_size = 768
        
    def forward(self, x):
        # ViT 前向传播
        outputs = self.vit(pixel_values=x)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 提取 [CLS] token 的输出
        logits = self.classifier(cls_output)
        return logits

