import torch
import torch.nn as nn

class SimpleAttentionHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.query = nn.Linear(in_ch, in_ch)
        self.key   = nn.Linear(in_ch, in_ch)
        self.value = nn.Linear(in_ch, in_ch)
        self.fc    = nn.Linear(in_ch, num_classes)

    def forward(self, feats: torch.Tensor):
        # feats: [B, C]
        q = self.query(feats)
        k = self.key(feats)
        v = self.value(feats)
        attn = torch.softmax((q*k)/ (feats.shape[-1]**0.5), dim=-1)
        z = attn * v
        return self.fc(z)
