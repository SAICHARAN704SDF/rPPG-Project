import torch
import torch.nn as nn
import torch.nn.functional as F

class RhythmMamba(nn.Module):
    def __init__(self, d_model=128, n_layers=2):
        super().__init__()
        
        # 1. Spatial Encoder (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, d_model, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 2. Temporal modeling (Pure PyTorch Rhythm Layers)
        self.temporal_layers = nn.ModuleList([
            nn.GRU(d_model, d_model // 2, batch_first=True, bidirectional=True)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [Batch, Time, Channels, H, W]
        B, T, C, H, W = x.shape
        
        # Flatten batch + time to process through CNN
        x = x.view(B * T, C, H, W)
        x = self.cnn(x).view(B, T, -1) 
        
        # Process through temporal rhythm layers
        for layer in self.temporal_layers:
            out, _ = layer(x)
            x = x + out # Residual connection
            
        x = self.norm(x)
        return self.output(x).squeeze(-1)
