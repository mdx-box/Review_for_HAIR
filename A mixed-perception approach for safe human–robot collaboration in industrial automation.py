import torch
import torch.nn as nn

class HumanActionRecognition3DCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(HumanActionRecognition3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 96, kernel_size=(7, 7, 7), stride=1, padding=(3, 3, 3))  # 3x180x320x3 -> 96x180x320x3
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=3, stride=3)  # 96x60x106x1
        self.conv2 = nn.Conv3d(96, 256, kernel_size=(5, 5, 5), stride=1, padding=(2, 2, 2))  # 256x60x106x1
        self.conv3 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))  # 256x60x106x1
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))  # 512x60x106x1
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 512x30x53x1
        self.flatten_dim = 512 * 30 * 53 * 1
        self.fc1 = nn.Linear(self.flatten_dim, 2024)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(2024, 1024)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x
