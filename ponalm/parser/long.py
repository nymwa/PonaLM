from ponalm.generation.long import long_main
from .model import parse_model_args

def parse_args(parser):
    parser.add_argument('--max-len', type = int, default = 1000)


def long(first):
    parser = first.add_parser('long')
    parse_args(parser)
    parse_model_args(parser)
    parser.set_defaults(handler = long_main)

