import torch
import torch.nn as nn

class WordDropout(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            x.masked_fill_(torch.rand(*x.shape[: -1], 1, device = x.device) < self.p, 0)
        return x


class PonaLMEmbedding(nn.Module):

    def __init__(self, d_vocab, d_model, dropout, word_dropout, padding_idx = 0):
        super().__init__()
        self.token_embedding = nn.Embedding(d_vocab, d_model, padding_idx = padding_idx)
        self.word_dropout = WordDropout(word_dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.word_dropout(x)
        x = self.dropout(x)
        x = self.norm(x)
        return x

