import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseCNN_LSTM_Attn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # --- CNN feature extractor over joints ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(1, 5), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(64)
        
        # CHANGED: 128 -> 64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d((1, 2))  
        self.dropout = nn.Dropout(0.3)
        
        # --- Temporal convolution over time ---
        # CHANGED: 128 -> 64
        self.temp_conv = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))
        self.bn_temp = nn.BatchNorm2d(64)
        
        # --- LSTM for temporal modeling ---
        # CHANGED: input_size reflects channel change, hidden_size 128 -> 256
        self.lstm = nn.LSTM(
            input_size=64 * (47 // 2), 
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        # --- Attention layer ---
        # CHANGED: input is 256 * 2 (for bidirectional) -> 512
        self.attn = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # --- Classifier ---
        # CHANGED: input 512, hidden 512, then to classes
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 256), # This matches your fc.1.weight mismatch
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
