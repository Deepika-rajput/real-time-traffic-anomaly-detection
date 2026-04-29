import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class AccidentTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # ✅ Updated pretrained loading
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()

        # ✅ batch_first fix
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=2
        )

        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)
        features = self.cnn(x)

        features = features.view(B, T, 512)   # no permute needed

        out = self.transformer(features)
        out = out.mean(dim=1)   # average over time

        return self.fc(out)