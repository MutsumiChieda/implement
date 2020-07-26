import torch
from torch import nn
from self_attention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, k, heads, mask=False):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k))

        self.mask = mask

    def forward(self, x):
        attended = self.attention(x, self.mask)
        x = self.norm1(attended + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)