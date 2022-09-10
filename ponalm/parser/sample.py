from ponalm.generation.sample import sample_main
from .model import parse_model_args

def parse_args(parser):
    parser.add_argument('--iters', type = int, default = 10)
    parser.add_argument('--prefix', default = None)
    parser.add_argument('--terminate-quot', action = 'store_true')


def sample(first):
    parser = first.add_parser('sample')
    parse_args(parser)
    parse_model_args(parser)
    parser.set_defaults(handler = sample_main)

