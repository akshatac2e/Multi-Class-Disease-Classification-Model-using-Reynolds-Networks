import torch
import torch.nn as nn
from ..reynolds.operators import ReynoldsOperator

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.net(x) + x)

class ReynoldsNetTiny(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResidualBlock(32)
        self.rop1 = ReynoldsOperator(32, groups=4)
        self.down1  = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.block2 = ResidualBlock(64)
        self.rop2 = ReynoldsOperator(64, groups=8)
        self.down2  = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.block3 = ResidualBlock(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x); x = self.rop1(x)
        x = self.down1(x)
        x = self.block2(x); x = self.rop2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.head(x)
