import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim:int, heads: int):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5 # 缩放因子，用于缩放点积的结果
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1) # chunk的作用是将一个tensor分成几个tensor
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, t.shape[2] // self.heads), qkv)
        # 将q, k, v分别reshape成(batch_size, seq_len, heads, dim // heads)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale # 点积,
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        return self.to_out(out)


class GPT(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, seq_len, num_classes):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, dim))
        transformer_blocks = []