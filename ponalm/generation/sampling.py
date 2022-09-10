import numpy as np
import torch
from ponalm.batch import Batch

def top_p_sampling(logit, temperature, top_p):
    logit = logit / temperature
    probs = torch.softmax(logit, dim = -1)
    values, indices = torch.sort(probs)

    cumlated = torch.cumsum(values, -1)
    is_removed = cumlated < (1 - top_p)
    probs[indices[is_removed]] = 0

    probs = probs.cpu().numpy()
    probs = probs / sum(probs)
    next_token = np.random.choice(range(len(probs)), p = probs)
    return next_token


def calc_logit(model, vocab, last_token, hidden_list):
    inputs = torch.tensor([[last_token]]).T
    lengths = torch.tensor([1])
    batch = Batch(inputs, lengths = lengths)

    if torch.cuda.is_available():
        batch.cuda()

    with torch.no_grad():
        pred, hidden_list = model(batch, hidden_list = hidden_list)

    logit = pred[-1, 0, :]
    return logit, hidden_list

