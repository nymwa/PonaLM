from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from seriejo import Seriejo
from ponalm.vocab import load_vocab
from ponalm.dataset import Dataset
from ponalm.sampler import FixedSampler, RandomSampler
from ponalm.collator import Collator

from ponalm.train.model import get_lm_model
from ponalm.train.opter import Opter
from ponalm.train.losscalc import PonaLMLossCalc
from ponalm.train.trainer import Trainer

from logging import getLogger
logger = getLogger(__name__)

def load_dataset(name):
    data = Seriejo('data/{}'.format(name))
    dataset = Dataset(data)
    return dataset


def load_loaders(vocab, args):
    train_dataset = load_dataset('train')
    valid_dataset = load_dataset('valid')
    train_sampler = RandomSampler(train_dataset, args.max_tokens)
    valid_sampler = FixedSampler(valid_dataset, args.max_tokens)
    collator = Collator(vocab)
    train_loader = DataLoader(
            train_dataset,
            batch_sampler = train_sampler,
            collate_fn = collator)
    valid_loader = DataLoader(
            valid_dataset,
            batch_sampler = valid_sampler,
            collate_fn = collator)
    return train_loader, valid_loader

def train_main(args):
    print(args)

    vocab = load_vocab(args.vocab)
    train_loader, valid_loader = load_loaders(vocab, args)

    model = get_lm_model(vocab, args)
    model = model.cuda()
    logger.info('#params : {} ({})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    opter = Opter(
            model,
            args.lr,
            max_grad_norm = args.max_grad_norm,
            scheduler = args.scheduler,
            warmup_steps = args.warmup_steps,
            start_factor = args.start_factor,
            weight_decay = args.weight_decay)
    losscalc = PonaLMLossCalc(label_smoothing = args.label_smoothing)

    trainer = Trainer(
            train_loader,
            valid_loader,
            model,
            opter,
            losscalc,
            args.epochs,
            args.step_interval,
            args.save_interval)

    trainer.run()

