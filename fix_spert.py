import argparse
import torch
import random
import numpy as np

from args import train_argparser, eval_argparser, predict_argparser
from config_reader import process_configs
from spert import input_reader
from spert.fix_trainer import SpERTTrainer

def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    print("seed set")

def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __train(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, unlabeled_path=run_args.unlabeled_path, input_reader_cls=input_reader.AugInputReader)


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


def __eval(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


def _predict():
    arg_parser = predict_argparser()
    process_configs(target=__predict, arg_parser=arg_parser)


def __predict(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.predict(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                    input_reader_cls=input_reader.JsonPredictionInputReader)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()
    # setseed(42)

    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    elif args.mode == 'predict':
        _predict()
    else:
        raise Exception("Mode not in ['train', 'eval', 'predict'], e.g. 'python spert.py train ...'")
