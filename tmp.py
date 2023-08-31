from header import *
from dataloader import *
from models import *
from config import *
import sys, os, datetime
import torch.multiprocessing as mp
from time import time

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='wikipedia', type=str)
    parser.add_argument('--model', default='copyisallyouneed', type=str)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--loss_type', type=str, default='focal_loss')
    parser.add_argument('--version', type=str)
    parser.add_argument('--pretrain_model_path', type=str)
    parser.add_argument('--training_data_dir', type=str, default='/apdcephfs/share_916081/ponybwcao/phrase_extraction/data/minipile/match_result_tok')
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--total_step', type=int, default=100000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--prebatch_step', type=int, default=5)
    parser.add_argument('--total_workers', type=int)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--random_initialize', type=bool, default=False)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--data_file_num', type=int, default=8)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.7)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--iter_to_accumulate', type=int, default=1)
    parser.add_argument('--warmup_step', type=int, default=0)
    parser.add_argument('--prebatch_num', type=int, default=1000)
    parser.add_argument('--phrase_dim', type=int, default=128)
    return parser.parse_args()


def main(**args):
    torch.cuda.empty_cache()

    config = load_config(args)
    args.update(config)

    agent = load_model(args)
    result = agent.model.evaluate(quiet=False)
    print(result)


if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
