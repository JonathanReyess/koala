"""PoseCNN_LSTM_Attn model definition.

This is the single source of truth for train.py/evaluate.py. It is built
directly from src/backend/model.py (the architecture that actually loads
`src/backend/best_model.pt` successfully: 64-channel CNN, 256-unit LSTM) --
NOT from the pre-tuning 128-channel version that used to live in this
package and silently failed to load the real weights (see AUDIT.md §9,
where the previous models/ package was deleted for exactly this reason).

IMPORTANT: if you change this architecture, `src/backend/model.py` (the
deployed copy) will no longer match a checkpoint trained here unless you
update it too. They are intentionally kept as two separate files rather than
one shared import, so the FastAPI backend stays deployable on its own
without needing the rest of this repo on its PYTHONPATH.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Values Optuna converged on in notebook/KSL.ipynb (Cell 8's `best_params`),
# reproduced here as the CLI defaults for train.py.
DEFAULT_CNN_HIDDEN = 64
DEFAULT_LSTM_HIDDEN = 256
DEFAULT_DROPOUT_RATE = 0.2224530595939839
DEFAULT_LR = 0.0003421517873280485

NUM_JOINTS = 47


class PoseCNN_LSTM_Attn(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cnn_hidden: int = DEFAULT_CNN_HIDDEN,
        lstm_hidden: int = DEFAULT_LSTM_HIDDEN,
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(1, 5), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, cnn_hidden, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(cnn_hidden)

        self.pool = nn.MaxPool2d((1, 2))
        self.dropout = nn.Dropout(dropout_rate)

        self.temp_conv = nn.Conv2d(cnn_hidden, cnn_hidden, kernel_size=(3, 1), padding=(1, 0))
        self.bn_temp = nn.BatchNorm2d(cnn_hidden)

        self.lstm = nn.LSTM(
            input_size=cnn_hidden * (NUM_JOINTS // 2),
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attn = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1),
        )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 3, 32, 47)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn_temp(self.temp_conv(x)))

        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, C, J)
        x = x.view(x.size(0), x.size(1), -1)  # (B, T, C*J)

        out, _ = self.lstm(x)  # (B, T, lstm_hidden*2)

        attn_scores = self.attn(out)  # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * out, dim=1)  # (B, lstm_hidden*2)

        return self.fc(context)
