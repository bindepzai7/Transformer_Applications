import torch
import torch.nn as nn

from src.Encoder import TransformerEncoder
from src.Decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, max_len, num_layers, num_heads, ff_dim, dropout=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = TransformerEncoder(src_vocab_size, embed_dim, max_len, num_layers, num_heads, ff_dim, dropout, device)
        self.decoder = TransformerDecoder(tgt_vocab_size, embed_dim, max_len, num_layers, num_heads, ff_dim, dropout, device)
        self.fc_out = nn.Linear(embed_dim, tgt_vocab_size)

    def generate_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).bool()

        mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=self.device), diagonal=1)
        mask = mask == 1  # Convert to boolean mask

        tgt_mask = mask.float().masked_fill(mask == 0, float(0.0)).masked_fill(mask == 1, float('-inf'))

        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        enc_output = self.encoder(src)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc_out(dec_output)
        return output