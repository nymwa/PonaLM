import numpy as np
from ponalm.vocab import load_vocab
from ponalm.train.model import get_lm_model

from ponalm.preproc.preproc import LMPreproc
from ponalm.preproc.postproc import LMPostproc
from ponalm.generation.sampler import SentenceSampler


def prepare_prefix(args, vocab):
    preproc = LMPreproc()
    if args.prefix is not None:
        prefix = preproc(args.prefix)
        prefix = [vocab(token) for token in prefix.split()]
    else:
        prefix = None
    return prefix


def prepare_terminal(args, vocab):
    if args.terminate_quot:
        terminal = {vocab('"')}
    else:
        terminal = None
    return terminal


def sample_main(args):
    vocab = load_vocab(args.vocab)
    prefix = prepare_prefix(args, vocab)
    terminal = prepare_terminal(args, vocab)

    model = get_lm_model(vocab, args)
    model.eval()

    postproc = LMPostproc()
    sampler = SentenceSampler(vocab, model)

    for i in range(args.iters):
        sent, probs = sampler(sent = prefix, terminal = terminal)
        sent = ' '.join([vocab[x] for x in sent])
        sent = postproc(sent)
        print(np.log(probs).sum(), '\t', sent)

