import numpy as np
import torch

from ponalm.vocab import load_vocab
from ponalm.train.model import get_lm_model

from seriejo import Seriejo
from ponalm.dataset import Dataset

from ponalm.batch import Batch

from logging import getLogger
logger = getLogger(__name__)

def load_dataset(name):
    data = Seriejo(name)
    dataset = Dataset(data)
    logger.info('dataset <- {}'.format(name))
    return dataset


def make_batch(vocab, sent):
    inputs = torch.tensor([[vocab.bos] + sent]).T
    lengths = torch.tensor([inputs.shape[0]])
    batch = Batch(inputs, lengths = lengths)
    if torch.cuda.is_available():
        batch.cuda()
    return batch


def calc_probs(model, vocab, sent):
    batch = make_batch(vocab, sent)

    with torch.no_grad():
        pred, _ = model(batch)

    probs = torch.softmax(pred, dim = -1)
    return probs


def make_prob_list(model, vocab, dataset):
    for sent in dataset:
        probs = calc_probs(model, vocab, sent)
        for index, prob in zip(sent + [vocab.eos], probs):
            yield prob[0][index].item()


def ppl_main(args):
    vocab = load_vocab(args.vocab)
    model = get_lm_model(vocab, args)
    model.eval()
    dataset = load_dataset(args.dataset)

    probs = [prob for prob in make_prob_list(model, vocab, dataset)]
    probs = [-np.log2(prob) for prob in probs]
    ppl = 2 ** np.mean(probs)
    logger.info('ppl: {:.4f}'.format(ppl))

