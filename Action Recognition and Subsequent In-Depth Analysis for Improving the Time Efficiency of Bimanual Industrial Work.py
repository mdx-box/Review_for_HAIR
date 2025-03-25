import torch
import torch.nn as nn

class CBiLSTM_MO(nn.Module):
    def __init__(self, input_size=14, hidden_size=128, num_classes=7, num_coord_layers=1):
        """
        C-BiLSTM+MO 网络
        - input_size: 每个手的输入特征维度 (6关节×2D + MOscore + MOscore_diff = 14)
        - hidden_size: 隐藏层维度 (文献中为128)
        - num_classes: 输出类别数 (pick-and-place: 7, assembly: 11)
        - num_coord_layers: 协调流的层数 (假设为1)
        """
        super(CBiLSTM_MO, self).__init__()
        self.hidden_size = hidden_size

        # 双手各自的 BiLSTM 流
        self.right_bilstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.left_bilstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)

        # 协调流 BiLSTM (输入为双手的隐藏状态拼接)
        self.coord_bilstm = nn.LSTM(hidden_size * 4, hidden_size, num_layers=num_coord_layers, 
                                    batch_first=True, bidirectional=True)  # 4 = 2方向 × 2手

        # 输出层
        self.fc_right = nn.Linear(hidden_size * 2, num_classes)  # 双向输出为 2×hidden_size
        self.fc_left = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x_right, x_left):
        """
        - x_right: 右手输入序列，形状 (batch_size, seq_len, input_size)
        - x_left: 左手输入序列，形状 (batch_size, seq_len, input_size)
        返回：左右手的动作标签序列，形状 (batch_size, seq_len, num_classes)
        """
        batch_size, seq_len, _ = x_right.size()

        # 双手各自的 BiLSTM 处理
        right_out, _ = self.right_bilstm(x_right)  # (batch_size, seq_len, hidden_size * 2)
        left_out, _ = self.left_bilstm(x_left)    # (batch_size, seq_len, hidden_size * 2)

        # 拼接双手输出用于协调流
        coord_in = torch.cat((right_out, left_out), dim=-1)  # (batch_size, seq_len, hidden_size * 4)
        coord_out, _ = self.coord_bilstm(coord_in)           # (batch_size, seq_len, hidden_size * 2)

        # 融合协调流输出与原始 BiLSTM 输出
        right_combined = right_out + coord_out  # 残差连接
        left_combined = left_out + coord_out

        # 输出分类
        right_logits = self.fc_right(right_combined)  # (batch_size, seq_len, num_classes)
        left_logits = self.fc_left(left_combined)     # (batch_size, seq_len, num_classes)

        return right_logits, left_logits
