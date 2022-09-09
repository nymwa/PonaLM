from ponalm.model.lm import PonaLM

def get_lm_model(vocab, args):
    model = PonaLM(
            len(vocab),
            args.hidden_dim,
            args.dropout,
            args.word_dropout,
            args.num_layers,
            padding_idx = vocab.pad)

    if not args.no_share_embedding:
        # なぜか式の左右を逆にすると動かない
        # パラメータの初期化の問題かもしれないが謎
        model.embedding.token_embedding.weight = model.fc.weight

    return model

