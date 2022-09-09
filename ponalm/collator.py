import random as rd
import torch
from torch.nn.utils.rnn import pad_sequence as pad
from ponalm.batch import Batch

class Collator:

    def __init__(self, vocab):
        self.vocab = vocab

    def make_tensors(self, batch):
        inputs  = [torch.tensor([self.vocab.bos] + sent) for sent in batch]
        outputs = [torch.tensor(sent + [self.vocab.eos]) for sent in batch]
        lengths = [len(sent) + 1 for sent in batch]

        inputs = pad(inputs, padding_value = self.vocab.pad)
        outputs = pad(outputs, padding_value = -100)
        return inputs, outputs, lengths

    def __call__(self, batch):
        inputs, outputs, lengths = self.make_tensors(batch)
        return Batch(inputs, outputs, lengths)

