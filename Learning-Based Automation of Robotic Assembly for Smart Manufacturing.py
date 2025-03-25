import torch
import torch.nn as nn

class AssemblyActionRecognition3DCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(AssemblyActionRecognition3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 96, kernel_size=(7, 7, 3), stride=(2, 2, 1), padding=(3, 3, 1))  # 3x1080x1920x10 -> 96x539x959x10
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 3, 2), stride=(2, 2, 2))  # 96x269x479x5
        self.conv2 = nn.Conv3d(96, 256, kernel_size=(5, 5, 3), stride=1, padding=(2, 2, 1))  # 256x269x479x5
        self.maxpool2 = nn.MaxPool3d(kernel_size=(3, 3, 2), stride=(2, 2, 2))  # 256x134x239x2
        self.conv3 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))  # 512x134x239x2
        self.conv4 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))  # 512x134x239x2
        self.conv5 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))  # 512x134x239x2
        self.maxpool3 = nn.MaxPool3d(kernel_size=(3, 3, 2), stride=(2, 2, 2))  # 512x67x119x1
        self.flatten_dim = 512 * 67 * 119 * 1
        self.fc1 = nn.Linear(self.flatten_dim, 4096)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4096, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
