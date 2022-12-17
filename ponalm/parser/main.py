from argparse import ArgumentParser
from .preproc import preproc
from .train import train
from .sample import sample
from .ppl import ppl
from .long import long

def parse_args():
    parser = ArgumentParser()
    first = parser.add_subparsers()

    preproc(first)
    train(first)
    sample(first)
    ppl(first)
    long(first)

    return parser.parse_args()

