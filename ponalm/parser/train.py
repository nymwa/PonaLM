from ponalm.train.main import train_main

def parse_args(parser):
    parser.add_argument('--vocab', default = 'vocab.txt')
    parser.add_argument('--max-tokens', type = int, default = 4000)

    parser.add_argument('--hidden-dim', type = int, default = 512)
    parser.add_argument('--num-layers', type = int, default = 6)

    parser.add_argument('--dropout', type = float, default = 0.3)
    parser.add_argument('--word-dropout', type = float, default = 0.3)
    parser.add_argument('--no-share-embedding', action = 'store_true')

    parser.add_argument('--label-smoothing', type = float, default = 0.0)
    parser.add_argument('--lr', type = float, default = 0.0001)
    parser.add_argument('--weight-decay', type = float, default = 0.01)
    parser.add_argument('--max-grad-norm', type = float, default = 1.0)
    parser.add_argument('--scheduler', default = 'linexp')
    parser.add_argument('--warmup-steps', type = int, default = 4000)
    parser.add_argument('--start-factor', type = float, default = 1.0)

    parser.add_argument('--epochs', type = int, default = 200)
    parser.add_argument('--step-interval', type = int, default = 1)
    parser.add_argument('--save-interval', type = int, default = 10)


def train(first):
    parser = first.add_parser('train')
    parse_args(parser)
    parser.set_defaults(handler = train_main)

