import sys
import logging
from ponalm.vocab import load_vocab
from ponalm.train.model import get_lm_model

from ilonimi import Normalizer
from ponalm.preproc.postproc import LMPostproc
from ponalm.generation.token_sampler import TokenSampler


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


def long_main(args):
    logging.getLogger().setLevel(logging.ERROR)
    vocab = load_vocab(args.vocab)
    model = get_lm_model(vocab, args)
    model.eval()

    sampler = TokenSampler(
            vocab, model,
            index = 0,
            last_token = vocab.bos,
            hidden_list = None,
            temperature = 1.0,
            top_p = 0.8,
            max_tokens = args.max_len,
            stop_ratio = 0.3,
            beta = 1.0,
            terminal = None,
            min_len = args.max_len)

    postproc = LMPostproc()
    normalizer = Normalizer()
    sent = []
    line = ''
    for _ in range(args.max_len):
        sent.append(sampler())
        new_line = normalizer(postproc(' '.join([vocab[x] for x in sent])))
        if ' ' in new_line[len(line):]:
            print(new_line[len(line):], end = '')
            sys.stdout.flush()
            line = new_line

