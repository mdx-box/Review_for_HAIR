import torch
import torch.nn as nn

# 自定义激活函数
def hardsigmoid(x):
    """Hardsigmoid激活函数"""
    return torch.clamp(0.2 * x + 0.5, 0, 1)

def softsign(x):
    """Softsign激活函数"""
    return x / (1 + torch.abs(x))

class ILSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        改进的LSTM单元
        - input_size: 输入特征维度
        - hidden_size: 隐藏层维度
        """
        super(ILSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 输入门、遗忘门、输出门和候选状态的线性变换
        self.W_xi = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_xf = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_xo = nn.Linear(input_size, hidden_size, bias=True)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_xc = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hc = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, h_prev, c_prev):
        """
        - x: 输入，形状 (batch_size, input_size)
        - h_prev: 前一时刻隐藏状态，形状 (batch_size, hidden_size)
        - c_prev: 前一时刻细胞状态，形状 (batch_size, hidden_size)
        返回：当前隐藏状态 h_t 和细胞状态 c_t
        """
        # 输入门
        i_t = hardsigmoid(self.W_xi(x) + self.W_hi(h_prev))
        # 遗忘门
        f_t = hardsigmoid(self.W_xf(x) + self.W_hf(h_prev))
        # 输出门
        o_t = hardsigmoid(self.W_xo(x) + self.W_ho(h_prev))
        # 候选细胞状态
        c_tilde = softsign(self.W_xc(x) + self.W_hc(h_prev))
        # 当前细胞状态
        c_t = f_t * c_prev + i_t * c_tilde
        # 当前隐藏状态
        h_t = o_t * softsign(c_t)
        return h_t, c_t

class IBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        改进的双向LSTM层
        - input_size: 输入特征维度
        - hidden_size: 隐藏层维度
        """
        super(IBiLSTM, self).__init__()
        self.forward_cell = ILSTMCell(input_size, hidden_size)
        self.backward_cell = ILSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        - x: 输入序列，形状 (batch_size, seq_len, input_size)
        返回：输出序列，形状 (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        h_forward = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_forward = torch.zeros(batch_size, self.hidden_size, device=x.device)
        h_backward = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_backward = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        forward_out = []
        backward_out = []
        
        # 前向传递
        for t in range(seq_len):
            h_forward, c_forward = self.forward_cell(x[:, t, :], h_forward, c_forward)
            forward_out.append(h_forward.unsqueeze(1))
        
        # 后向传递
        for t in reversed(range(seq_len)):
            h_backward, c_backward = self.backward_cell(x[:, t, :], h_backward, c_backward)
            backward_out.insert(0, h_backward.unsqueeze(1))
        
        # 拼接前向和后向输出
        forward_out = torch.cat(forward_out, dim=1)
        backward_out = torch.cat(backward_out, dim=1)
        output = hardsigmoid(forward_out + backward_out)  # 结合前向和后向
        return output

class HybridRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_bilstm_layers=2):
        """
        混合RNN架构：多层IBi-LSTM + 1层ILSTM
        - input_size: 输入特征维度（骨架关节位置）
        - hidden_size: 隐藏层维度
        - num_bilstm_layers: IBi-LSTM层数
        """
        super(HybridRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # IBi-LSTM层
        self.bilstm_layers = nn.ModuleList([
            IBiLSTM(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_bilstm_layers)
        ])
        
        # ILSTM层
        self.lstm_layer = ILSTMCell(hidden_size, hidden_size)
        
        # 输出层（预测下一帧骨架位置）
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """
        - x: 输入序列，形状 (batch_size, seq_len, input_size)
        返回：预测序列，形状 (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # 通过多层IBi-LSTM
        out = x
        for bilstm in self.bilstm_layers:
            out = bilstm(out)
        
        # 通过ILSTM层
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            h_t, c_t = self.lstm_layer(out[:, t, :], h_t, c_t)
            outputs.append(self.fc(h_t).unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        return output