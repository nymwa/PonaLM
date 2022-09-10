import torch
from ponalm.batch import Batch
from .token_sampler import TokenSampler

class SentenceSampler:

    def __init__(self, vocab, model):
        self.vocab = vocab
        self.model = model

    def make_hidden_list(self, sent):
        inputs = torch.tensor([[self.vocab.bos] + sent[:-1]]).T
        lengths = torch.tensor([inputs.shape[0]])
        batch = Batch(inputs, lengths = lengths)

        if torch.cuda.is_available():
            batch.cuda()

        with torch.no_grad():
            _, hidden_list = self.model(batch)
        return hidden_list

    def __call__(
            self,
            temperature = 1.0,
            top_p = 0.8,
            max_tokens = 128,
            stop_ratio = 0.3,
            beta = 1.0,
            sent = None,
            terminal = None,
            min_len = None):

        if sent is None:
            sent = []
            last_token = self.vocab.bos
            hidden_list = None
        else:
            sent = sent[:]
            last_token = sent[-1]
            hidden_list = self.make_hidden_list(sent)

        sampler = TokenSampler(
                self.vocab, self.model,
                index = len(sent),
                last_token = last_token,
                hidden_list = hidden_list,
                temperature = temperature,
                top_p = top_p,
                max_tokens = max_tokens,
                stop_ratio = stop_ratio,
                beta = beta,
                min_len = min_len,
                terminal = terminal)

        while sampler.index <= max_tokens:
            next_token = sampler()
            if next_token == self.vocab.eos:
                break
            sent.append(next_token)
            if (terminal is not None) and (next_token in terminal):
                break

        return sent

