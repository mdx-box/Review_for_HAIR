import torch
import torch.nn as nn
import torchvision.models as models

# 1. 定义两流 ResNet-152 特征提取器
class TwoStreamResNet(nn.Module):
    def __init__(self):
        super(TwoStreamResNet, self).__init__()
        # 空间流：RGB 输入
        self.spatial_stream = models.resnet152(pretrained=True)
        self.spatial_stream.fc = nn.Identity()  # 移除最后一层全连接层，输出 2048 维特征
        # 时间流：光流输入
        self.temporal_stream = models.resnet152(pretrained=True)
        self.temporal_stream.fc = nn.Identity()  # 输出 2048 维特征

    def forward(self, rgb, flow):
        spatial_features = self.spatial_stream(rgb)  # [batch, 10, 2048]
        temporal_features = self.temporal_stream(flow)  # [batch, 10, 2048]
        # 拼接空间和时间特征
        features = torch.cat((spatial_features, temporal_features), dim=-1)  # [batch, 10, 4096]
        return features

# 2. 定义堆叠 LSTM 模块
class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(StackedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 100)  # 输出 100 维特征向量

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        out = self.dropout(out[:, -1, :])  # 取最后一个时间步输出
        out = self.fc(out)  # [batch, 100]
        return out

# 3. 定义完整动作预测网络
class ActionForecastingNetwork(nn.Module):
    def __init__(self, num_classes, progress_steps=[5, 10, 20]):
        super(ActionForecastingNetwork, self).__init__()
        # 两流特征提取器
        self.feature_extractor = TwoStreamResNet()
        
        # 局部预测模块 (LSTM-act)
        self.local_lstm = StackedLSTM(input_size=4096, hidden_size=32, num_layers=2)
        self.local_fc = nn.Linear(100, num_classes)  # 输出动作类别
        
        # 全局进度估计模块 (LSTM-5, LSTM-10, LSTM-20)
        self.global_lstms = nn.ModuleList([
            StackedLSTM(input_size=4096, hidden_size=32, num_layers=2) for _ in progress_steps
        ])
        self.global_fcs = nn.ModuleList([
            nn.Linear(100, step) for step in progress_steps  # 每个粒度对应一个输出
        ])
        
        # 联合特征融合
        self.combined_fc = nn.Linear(100 * (1 + len(progress_steps)), num_classes)

    def forward(self, rgb, flow):
        # 提取特征
        features = self.feature_extractor(rgb, flow)  # [batch, 10, 4096]
        
        # 局部预测
        local_features = self.local_lstm(features)  # [batch, 100]
        local_pred = self.local_fc(local_features)  # [batch, num_classes]
        
        # 全局进度估计
        global_features = []
        global_preds = []
        for lstm, fc in zip(self.global_lstms, self.global_fcs):
            feat = lstm(features)  # [batch, 100]
            pred = fc(feat)  # [batch, progress_step]
            global_features.append(feat)
            global_preds.append(pred)
        
        # 联合特征
        combined_features = torch.cat([local_features] + global_features, dim=-1)  # [batch, 400]
        combined_pred = self.combined_fc(combined_features)  # [batch, num_classes]
        
        return local_pred, global_preds, combined_pred