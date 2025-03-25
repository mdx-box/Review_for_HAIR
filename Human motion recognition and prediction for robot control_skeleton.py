import torch
import torch.nn as nn

class EnhancedRNN(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=50, dense_dim=150, output_dim=3):
        super(EnhancedRNN, self).__init__()
        # Component Unit (e.g., "arm")
        self.rnn_component = nn.RNN(input_dim, hidden_dim, batch_first=True)
        # Coordination Unit (e.g., "arm-spine")
        self.rnn_coordination = nn.RNN(input_dim, hidden_dim, batch_first=True)
        # Dense Layers
        self.dense1 = nn.Linear(hidden_dim * 2, dense_dim)  # Concatenate outputs from both units
        self.dense2 = nn.Linear(dense_dim, dense_dim)
        self.output_layer = nn.Linear(dense_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Component Unit forward
        component_out, _ = self.rnn_component(x)  # [batch, 30, 50]
        component_out = component_out[:, -1, :]  # Take the last time step: [batch, 50]
        # Coordination Unit forward
        coordination_out, _ = self.rnn_coordination(x)  # [batch, 30, 50]
        coordination_out = coordination_out[:, -1, :]  # [batch, 50]
        # Concatenate outputs
        combined = torch.cat((component_out, coordination_out), dim=1)  # [batch, 100]
        # Dense Layers
        x = self.relu(self.dense1(combined))  # [batch, 150]
        x = self.relu(self.dense2(x))  # [batch, 150]
        x = self.output_layer(x)  # [batch, 3]
        return x

