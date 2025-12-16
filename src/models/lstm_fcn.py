import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMFCNClassifier(nn.Module):
    def __init__(self, num_classes, in_channels=1, lstm_hidden=128, dropout=0.3):
        super().__init__()

        # FCN branch (Conv + BN helps a lot)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)

        # LSTM branch: project 1 -> 32 dims first (critical when F=1)
        self.lstm_in = nn.Linear(in_channels, 32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=lstm_hidden, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128 + lstm_hidden, num_classes)

    def forward(self, x):
        # x: [B,T,1]
        # FCN branch: [B,1,T]
        z = x.permute(0, 2, 1)
        z = F.relu(self.bn1(self.conv1(z)))
        z = F.relu(self.bn2(self.conv2(z)))
        z = F.relu(self.bn3(self.conv3(z)))
        z = z.mean(dim=2)  # GAP -> [B,128]

        # LSTM branch
        x_l = self.lstm_in(x)         # [B,T,32]
        _, (h_n, _) = self.lstm(x_l)
        h = h_n[-1]                   # [B,H]

        out = torch.cat([z, h], dim=1)
        out = self.dropout(out)
        return self.fc(out)
