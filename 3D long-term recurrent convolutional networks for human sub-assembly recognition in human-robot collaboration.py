import torch
import torch.nn as nn

class LRCN3D(nn.Module):
    def __init__(self, num_classes=7, lstm_hidden_size=256):
        super(LRCN3D, self).__init__()
        
        # 3D CNN部分
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=3)  # 输出: [32, 34, 22, 22]
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=3)  # 输出: [64, 12, 8, 8]
        )
        
        # TimeDistributed Flatten和Dense层
        self.flatten = nn.Flatten(start_dim=2)  # 将空间维度展平
        self.dense1 = nn.Linear(64*8*8, 4096)  # Flatten后接4096维全连接层
        self.dense2 = nn.Linear(4096, 512)     # 降维到512
        
        # LSTM部分
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        
    def forward(self, x):
        # 输入x: [batch_size, channels, frames, height, width]，如[batch_size, 3, 100, 64, 64]
        batch_size, C, T, H, W = x.size()
        
        # 3D CNN处理
        x = self.conv3d_1(x)  # [batch_size, 32, 34, 22, 22]
        x = self.conv3d_2(x)  # [batch_size, 64, 12, 8, 8]
        
        # 调整维度以适配TimeDistributed
        x = x.permute(0, 2, 1, 3, 4)  # [batch_size, 12, 64, 8, 8]
        x = self.flatten(x)    # [batch_size, 12, 64*8*8]
        
        # Dense层处理
        x = self.dense1(x)    # [batch_size, 12, 4096]
        x = nn.ReLU()(x)
        x = self.dense2(x)    # [batch_size, 12, 512]
        
        # LSTM处理时序特征
        lstm_out, (hn, cn) = self.lstm(x)  # [batch_size, 12, 256]
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # [batch_size, 256]
        
        # 全连接层输出分类结果
        out = self.fc(lstm_out)  # [batch_size, num_classes]
        
        return out

