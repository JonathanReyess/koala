import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseCNN_LSTM_Attn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # --- CNN feature extractor over joints ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(1, 5), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d((1, 2))  # reduce joint dimension
        self.dropout = nn.Dropout(0.3)
        
        # --- Temporal convolution over time ---
        self.temp_conv = nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0))
        self.bn_temp = nn.BatchNorm2d(128)
        
        # --- LSTM for temporal modeling ---
        self.lstm = nn.LSTM(
            input_size=128 * (47 // 2),
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        # --- Attention layer ---
        self.attn = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # --- Classifier ---
        self.fc = nn.Sequential(
            nn.BatchNorm1d(128 * 2),
            nn.Linear(128 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):  # x: (B, 3, 32, 47)
        # --- CNN over joints ---
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)               # (B, 128, 32, 23)
        x = self.dropout(x)
        
        # --- Temporal convolution ---
        x = F.relu(self.bn_temp(self.temp_conv(x)))  # (B, 128, 32, 23)
        
        # --- Prepare for LSTM ---
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, C, J)
        x = x.view(x.size(0), x.size(1), -1)    # (B, T, C*J)
        
        # --- LSTM ---
        out, _ = self.lstm(x)                   # (B, T, 256)
        
        # --- Attention pooling ---
        attn_scores = self.attn(out)            # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * out, dim=1)  # (B, 256)
        
        # --- Classification ---
        return self.fc(context)
