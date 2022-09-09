import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class PonaLMLayer(nn.Module):

    def __init__(self, d_model, dropout, nonlinearity = 'relu'):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.rnn = nn.RNN(d_model, d_model, nonlinearity = nonlinearity)
        self.norm = nn.LayerNorm(d_model)

    def forward_rnn(self, x, lengths, h, enforce_sorted):
        packed = pack(x, lengths, enforce_sorted = enforce_sorted)
        output, h = self.rnn(packed, h)
        x, _ = unpack(output)
        return x, h

    def forward_main(self, x, lengths, h, enforce_sorted):
        x = self.fc1(x)
        x = self.act(x)
        x, h = self.forward_rnn(x, lengths, h, enforce_sorted)
        x = self.fc2(x)
        return x, h

    def forward(self, x, lengths, h = None, enforce_sorted = True):
        z, h = self.forward_main(x, lengths, h, enforce_sorted)
        x = self.norm(x + z)
        return x, h

