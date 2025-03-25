import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectIdentificationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_objects):
        """
        第一个LSTM模块：识别手中的物体
        - input_size: 输入特征维度（手部关键点和物体检测特征）
        - hidden_size: LSTM隐藏层维度
        - num_layers: LSTM层数（文献中为2）
        - num_objects: 物体类别数（包括“无物体”）
        """
        super(ObjectIdentificationLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_objects)

    def forward(self, x):
        """
        - x: 输入序列，形状 (batch_size, seq_len, input_size)
        返回：物体类别概率，形状 (batch_size, seq_len, num_objects)
        """
        # LSTM处理序列
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        # 全连接层输出物体类别
        output = self.fc(lstm_out)  # (batch_size, seq_len, num_objects)
        return F.softmax(output, dim=-1)

class ActionRecognitionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_actions, num_objects):
        """
        第二个LSTM模块：装配动作识别
        - input_size: 输入特征维度（手部特征 + 物体信息 + 动作关系）
        - hidden_size: LSTM隐藏层维度
        - num_layers: LSTM层数（文献中为2）
        - num_actions: 动作类别数（包括“无动作”）
        - num_objects: 物体类别数（用于one-hot编码）
        """
        super(ActionRecognitionLSTM, self).__init__()
        self.num_objects = num_objects
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_actions)

    def forward(self, motion_features, object_probs, action_mask):
        """
        - motion_features: 手部运动特征，形状 (batch_size, seq_len, motion_size)
        - object_probs: 第一个LSTM输出的物体概率，形状 (batch_size, seq_len, num_objects)
        - action_mask: 动作关系掩码，形状 (batch_size, seq_len, num_actions)
        返回：动作类别概率，形状 (batch_size, seq_len, num_actions)
        """
        # 拼接输入：运动特征 + 物体信息 + 动作掩码
        batch_size, seq_len, _ = motion_features.size()
        input_data = torch.cat([motion_features, object_probs, action_mask], dim=-1)
        
        # 第一层LSTM
        lstm1_out, _ = self.lstm1(input_data)  # (batch_size, seq_len, hidden_size)
        # 第二层LSTM
        lstm2_out, _ = self.lstm2(lstm1_out)  # (batch_size, seq_len, hidden_size)
        # 全连接层输出动作类别
        output = self.fc(lstm2_out)  # (batch_size, seq_len, num_actions)
        return F.softmax(output, dim=-1)

class AssemblyActionNetwork(nn.Module):
    def __init__(self, motion_size, object_input_size, hidden_size, num_objects, num_actions):
        """
        完整的装配动作识别网络
        - motion_size: 手部运动特征维度
        - object_input_size: 物体检测输入特征维度
        - hidden_size: LSTM隐藏层维度
        - num_objects: 物体类别数
        - num_actions: 动作类别数
        """
        super(AssemblyActionNetwork, self).__init__()
        self.object_lstm = ObjectIdentificationLSTM(object_input_size, hidden_size, 2, num_objects)
        self.action_lstm = ActionRecognitionLSTM(motion_size + num_objects + num_actions, 
                                                 hidden_size, 2, num_actions, num_objects)

    def forward(self, motion_features, object_features, action_mask):
        """
        - motion_features: 手部运动特征，形状 (batch_size, seq_len, motion_size)
        - object_features: 物体检测特征，形状 (batch_size, seq_len, object_input_size)
        - action_mask: 动作关系掩码，形状 (batch_size, seq_len, num_actions)
        返回：动作类别概率，形状 (batch_size, seq_len, num_actions)
        """
        # 第一步：识别手中的物体
        object_probs = self.object_lstm(object_features)
        # 第二步：动作识别
        action_probs = self.action_lstm(motion_features, object_probs, action_mask)
        return action_probs
