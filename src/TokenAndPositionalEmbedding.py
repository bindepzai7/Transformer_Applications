import torch.nn as nn
import torch

class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, device):
        super().__init__()
        self.device = device
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.position_embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)
        
    def forward(self, x):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.token_embedding(x) + self.position_embedding(positions)
        return x
    
