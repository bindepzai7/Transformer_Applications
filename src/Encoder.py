import torch
import torch.nn as nn
from src.TokenAndPositionalEmbedding import TokenAndPositionalEmbedding

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim, bias=True)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        query = self.layernorm1(query + self.dropout1(attn_output))
        ffn_output = self.ffn(query)
        return self.layernorm2(query + self.dropout2(ffn_output))
    
class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, max_len, num_layers, num_heads, ff_dim, dropout=0.1, device='cpu'):
        super().__init__()
        self.embedding = TokenAndPositionalEmbedding(src_vocab_size, embed_dim, max_len, device)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, x, x)
        return x