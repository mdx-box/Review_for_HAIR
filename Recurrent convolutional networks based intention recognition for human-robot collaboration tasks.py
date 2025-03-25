import torch
import torch.nn as nn
import torchvision.models as models

class RCNNIntentionRecognition(nn.Module):
    def __init__(self, num_classes=9, lstm_hidden_size=128):
        super(RCNNIntentionRecognition, self).__init__()
        
        # DCNN部分：使用预训练的AlexNet
        self.dcnn = models.alexnet(pretrained=True)
        # 修改AlexNet的classifier部分，输出D维特征向量（假设D=4096，与AlexNet默认一致）
        self.dcnn.classifier = nn.Sequential(
            *list(self.dcnn.classifier.children())[:-1]  # 去掉最后一层，保留4096维输出
        )
        
        # LSTM部分
        self.lstm = nn.LSTM(input_size=4096, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        
    def forward(self, x):
        # 输入x: [batch_size, seq_len, channels, height, width]，这里seq_len=9, channels=3 (RGB)
        batch_size, seq_len, C, H, W = x.size()
        
        # DCNN处理每帧
        dcnn_out = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # [batch_size, channels, height, width]
            feat = self.dcnn(frame)   # [batch_size, 4096]
            dcnn_out.append(feat)
        dcnn_out = torch.stack(dcnn_out, dim=1)  # [batch_size, seq_len, 4096]
        
        # LSTM处理时序特征
        lstm_out, (hn, cn) = self.lstm(dcnn_out)  # lstm_out: [batch_size, seq_len, hidden_size]
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # 全连接层输出意图类别概率
        out = self.fc(lstm_out)  # [batch_size, num_classes]
        
        return out

