from ponalm.vocab import load_vocab
from ponalm.train.model import get_lm_model

from ponalm.preproc.postproc import LMPostproc
from ponalm.generation.token_sampler import TokenSampler

from tqdm import tqdm

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

    sent = []
    for _ in tqdm(range(args.max_len), bar_format = '{l_bar}{r_bar}'):
        sent.append(sampler())

    postproc = LMPostproc()
    sent = ' '.join([vocab[x] for x in sent])
    sent = postproc(sent)
    print(sent)

