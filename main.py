import argparse
from train import *
from utils import *


def mm(opt):
    set_seed(123)
    train_base(opt)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epochs', default=10, type=int,
                        help='Max training epoch')
    parser.add_argument('--dropout_prob', default=0.1, type=float,
                        help='drop out probability')
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate for the bert module')
    parser.add_argument('--other_lr', default=2e-4, type=float,
                        help='learning rate for the module except bert')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='max grad clip')
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0., type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--bert_dir', default="./pre_train", type=str)
    parser.add_argument('--train_path', default="./data/all.json", type=str)
    parser.add_argument('--dev_path', default="./data/dev.json", type=str)
    parser.add_argument('--eval_model', default=False, action='store_true',
                        help='whether to eval model after training')
    parser.add_argument('--output_dir', default="./checkpoint", type=str)
    parser.add_argument('--attack_train', default='', type=str,
                        help='fgm / pgd attack train when training')
    args = parser.parse_args()
    mm(args)
