import torch
import torch.nn as nn
import torchvision.models as models

class MRPNetwork(nn.Module):
    def __init__(self, num_classes=9, lstm_hidden_size=256):
        super(MRPNetwork, self).__init__()
        
        # CNN部分：使用预训练的VGG16，前13层卷积层
        vgg16 = models.vgg16(pretrained=True)
        self.cnn = nn.Sequential(*list(vgg16.features.children())[:13])  # 保留前13层
        # VGG16 features输出为512通道，需调整为LSTM输入
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # 输出固定为[batch_size, 512, 7, 7]
        self.flatten = nn.Flatten(start_dim=2)  # 展平为[batch_size, seq_len, 512*7*7]
        
        # LSTM部分
        self.lstm = nn.LSTM(input_size=512*7*7, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.dropout = nn.Dropout(p=0.8)  # Dropout比率0.8
        
    def forward(self, x):
        # 输入x: [batch_size, seq_len, channels, height, width]，如[batch_size, 5, 3, 160, 160]
        batch_size, seq_len, C, H, W = x.size()
        
        # CNN处理每帧
        cnn_out = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # [batch_size, channels, height, width]
            feat = self.cnn(frame)    # [batch_size, 512, H', W']
            feat = self.adaptive_pool(feat)  # [batch_size, 512, 7, 7]
            cnn_out.append(feat)
        cnn_out = torch.stack(cnn_out, dim=1)  # [batch_size, seq_len, 512, 7, 7]
        cnn_out = self.flatten(cnn_out)  # [batch_size, seq_len, 512*7*7]
        
        # LSTM处理时序特征
        lstm_out, (hn, cn) = self.lstm(cnn_out)  # [batch_size, seq_len, hidden_size]
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # 全连接层输出意图类别概率
        out = self.dropout(lstm_out)
        out = self.fc(out)  # [batch_size, num_classes]
        
        return out
