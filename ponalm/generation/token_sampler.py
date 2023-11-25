import torch
from .sampling import (
        top_p_sampling,
        calc_logit)


class TokenSampler:

    def __init__(
            self,
            vocab,
            model,
            index,
            last_token,
            hidden_list,
            temperature,
            top_p,
            max_tokens,
            stop_ratio,
            beta,
            terminal,
            min_len):

        self.vocab = vocab
        self.model = model
        self.index = index
        self.last_token = last_token
        self.hidden_list = hidden_list

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop_ratio = stop_ratio
        self.beta = beta
        self.terminal = terminal
        self.min_len = min_len

    def p_avoid_eos(self):
        return (self.min_len is not None) and (self.index < self.min_len)

    def postproc_logit(self, logit):
        logit[self.vocab.pad] = float('-inf')
        logit[self.vocab.bos] = float('-inf')
        logit[self.vocab.unk] = float('-inf')

        if self.p_avoid_eos():
            logit[self.vocab.eos] = float('-inf')
            if self.terminal is not None:
                for token in self.terminal:
                    logit[token] = float('-inf')

        if self.index > self.max_tokens * self.stop_ratio:
            logit[self.vocab.eos] += (self.index - self.max_tokens * self.stop_ratio) * self.beta
        return logit

    def __call__(self, output_prob = False):
        logit, self.hidden_list = calc_logit(
                self.model,
                self.vocab,
                self.last_token,
                self.hidden_list)
        logit = self.postproc_logit(logit)
        self.index += 1
        self.last_token = top_p_sampling(logit, self.temperature, self.top_p)

        if output_prob:
            probs = torch.softmax(logit, dim = -1)
            prob = float(probs[self.last_token])
            return self.last_token, prob
        else:
            return self.last_token

