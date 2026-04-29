import torch
import torch.nn as nn
import torchvision.models as models

class AccidentTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=2
        )

        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)
        features = self.cnn(x)

        features = features.view(B, T, 512)
        features = features.permute(1, 0, 2)

        out = self.transformer(features)
        out = out.mean(dim=0)

        return self.fc(out)