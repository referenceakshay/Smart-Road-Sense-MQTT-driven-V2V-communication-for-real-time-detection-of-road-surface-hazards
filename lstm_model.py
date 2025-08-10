import torch
import torch.nn as nn

class YOLOv8_LSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, num_layers=1, num_classes=2):
        super(YOLOv8_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out
