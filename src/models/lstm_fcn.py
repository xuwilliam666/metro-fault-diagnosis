import torch
import torch.nn as nn

class LSTMFCNClassifier(nn.Module):
    """
    Simplified LSTM-FCN model for time series classification used for testing the models.
    Input x should have shape (batch_size, seq_len, num_features)
    Output logits will have shape (batch_size, num_classes)
    """
    def __init__(self, num_classes, conv_channels=32, lstm_hidden=64):
        super().__init__()
        # 1D Convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.relu = nn.ReLU()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, batch_first=True, num_layers=1, bidirectional=False)

        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: [B, T, F]
        :return: logits
        """
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = x.permute(0, 2, 1)
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]
        logits = self.fc(h_last)
        return logits
