import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNStream(nn.Module):
    def __init__(self, input_shape):
        super(CNNStream, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)  # From Table 1
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)  # [B, 128, 6, 6]
        return x.view(x.size(0), -1)  # [B, 128*6*6]

class BiStreamCNN(nn.Module):
    def __init__(self, num_classes=31):  # 30 actions + 1 transition
        super(BiStreamCNN, self).__init__()
        self.ws_stream = CNNStream((192, 128))  # Workspace view
        self.obj_stream = CNNStream((230, 200))  # Object view
        self.fc1 = nn.Linear(128*6*6 * 2, 512)  # Concatenated features
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x_ws, x_obj):
        # x_ws: [B, 3, 192, 128], x_obj: [B, 3, 230, 200]
        ws_feat = self.ws_stream(x_ws)  # [B, 128*6*6]
        obj_feat = self.obj_stream(x_obj)  # [B, 128*6*6]
        feat = torch.cat([ws_feat, obj_feat], dim=1)  # [B, 128*6*6*2]
        out = F.relu(self.fc1(feat))
        out = self.fc2(out)  # [B, num_classes]
        return F.softmax(out, dim=1)

class VMM:
    def __init__(self, num_classes=31, max_length=26):
        self.num_classes = num_classes
        self.max_length = max_length
        self.transitions = {}  # {context: {action: count}}
        
    def train(self, sequences):
        # sequences: List[List[int]], e.g., [[0, 1, 2], [0, 3, 4]]
        for seq in sequences:
            for i in range(1, len(seq)):
                for j in range(max(0, i - self.max_length), i):
                    context = tuple(seq[j:i])
                    action = seq[i]
                    if context not in self.transitions:
                        self.transitions[context] = {}
                    self.transitions[context][action] = self.transitions[context].get(action, 0) + 1
        
        # Optimize context lengths using entropy
        self.optimal_contexts = {}
        for context in self.transitions:
            probs = self.get_transition_probs(context)
            entropy = -sum(p * (p.log() if p > 0 else 0) for p in probs)
            self.optimal_contexts[context] = entropy

    def get_transition_probs(self, context):
        if context not in self.transitions:
            return torch.ones(self.num_classes) / self.num_classes
        total = sum(self.transitions[context].values())
        probs = torch.zeros(self.num_classes)
        for action, count in self.transitions[context].items():
            probs[action] = count / total
        return probs

    def predict(self, sequence):
        # sequence: List[int], e.g., [0, 1, 2]
        for l in range(min(self.max_length, len(sequence)), 0, -1):
            context = tuple(sequence[-l:])
            if context in self.transitions:
                probs = self.get_transition_probs(context)
                return probs.argmax().item(), probs.max().item()
        return 0, 1.0 / self.num_classes  # Default if no context found


class HybridModel(nn.Module):
    def __init__(self, num_classes=31, max_length=26):
        super(HybridModel, self).__init__()
        self.cnn = BiStreamCNN(num_classes)
        self.vmm = VMM(num_classes, max_length)
        
    def forward(self, x_ws, x_obj):
        return self.cnn(x_ws, x_obj)
    
    def predict_next(self, sequence):
        return self.vmm.predict(sequence)
