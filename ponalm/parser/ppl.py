from ponalm.generation.ppl import ppl_main
from .model import parse_model_args

def parse_args(parser):
    parser.add_argument('--dataset', default = 'data/valid')


def ppl(first):
    parser = first.add_parser('ppl')
    parse_args(parser)
    parse_model_args(parser)
    parser.set_defaults(handler = ppl_main)

