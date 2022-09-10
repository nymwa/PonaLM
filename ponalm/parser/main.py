from argparse import ArgumentParser
from .preproc import preproc
from .train import train
from .sample import sample
from .ppl import ppl

def parse_args():
    parser = ArgumentParser()
    first = parser.add_subparsers()

    preproc(first)
    train(first)
    sample(first)
    ppl(first)

    return parser.parse_args()

