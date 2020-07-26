import torch
from torch import nn
import torch.nn.functional as F
from self_attention import SelfAttention
from transformer_block import TransformerBlock

class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, 
                 num_tokens, num_classes):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        # The sequence of transformer blocks 
        # that does all the heavy lifting
        tblocks = []
        for _ in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads, mask=True))
        self.tblocks = nn.Sequential(*tblocks)

        # Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, num_tokens)

    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        # generate token embeddings
        tokens = self.token_emb(x)
        b, t, k = tokens.size()

        # generate position embeddings
        positions = torch.arange(t)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.tblocks(x)

        # Average-pool over the t dimension and project 
        # to class probabilities
        x = self.toprobs(x.view(b*t, k)).view(b, t, self.num_tokens)
        return F.log_softmax(x, dim=2)