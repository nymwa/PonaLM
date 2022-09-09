from ponalm.log import init_logging
init_logging()
from ponalm.parser.main import parse_args

def main():
    args = parse_args()
    args.handler(args)

