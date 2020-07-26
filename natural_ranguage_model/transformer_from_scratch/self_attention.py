import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k, self.heads = k, heads
        # These compute the queries, keys and values 
        # for all heads
        # as a single concatenated vector
        self.tokeys    = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues  = nn.Linear(k, k * heads, bias=False)
        # This applies linear conversion to multi-head output 
        # to reduce the dimension to k
        self.unifyhead = nn.Linear(heads * k, k)

    def forward(self, x, mask=False):
        # Create queries, keys and values
        b,t,k = x.size()
        h = self.heads
        ## .view reshape (b, t, h*k) to (b,t,h,k)
        queries = self.toqueries(x).view(b,t,h,k)
        keys = self.tokeys(x).view(b,t,h,k)
        values = self.tovalues(x).view(b,t,h,k)

        # Compute the dot products
        ## Fold heads into the batch dimention
        ## .contiguous sorts index order in memory 
        ## to avoid error which happens 
        ## when applying .view to transposed tensor
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1,2).contiguous().view(b*h, t, k)
        ## Avoid gradient disappearing caused by softmax with large value input
        ## Divide by k**1/4 instead of k**1/2 to save memory
        queries = queries / (k**(1/4))
        keys = keys / (k**(1/4))
        ## dot has shape (b*h, t, t)
        dot = torch.bmm(queries, keys.transpose(1,2))
        if self.mask: # for text generation, hide forward part of sequence
            indices = torch.triu_indices(k,k,offset=0) # idx of upper triangle k x k
            dot[:,indices[0],indices[1]] = float('-inf')
        dot = F.softmax(dot, dim=2)

        # Apply the self attention to the values
        out = torch.bmm(dot,values).view(b,h,t,k)
        out = out.transpose(1,2).contiguous().view(b,t,h*k)
        return self.unifyhead(out)
        