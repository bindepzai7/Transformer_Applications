import torch 
import torch.nn as nn
from src.TokenAndPositionalEmbedding import TokenAndPositionalEmbedding

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output1, _ = self.attn1(x, x, x, attn_mask=tgt_mask)
        x = self.layernorm1(x + self.dropout1(attn_output1))
        
        attn_output2, _ = self.attn2(x, enc_output, enc_output, attn_mask=src_mask)
        x = self.layernorm2(x + self.dropout2(attn_output2))
        
        ffn_output = self.ffn(x)
        return self.layernorm3(x + self.dropout3(ffn_output))
    
class TransformerDecoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, max_len, num_layers, num_heads, ff_dim, dropout=0.1, device='cpu'):
        super().__init__()
        self.embedding = TokenAndPositionalEmbedding(tgt_vocab_size, embed_dim, max_len, device)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x