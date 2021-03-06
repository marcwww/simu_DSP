import nets
from macros import *
import torch
import utils
from hparams import opts
import argparse
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tasks import *
from torch.nn.init import orthogonal_, uniform_, xavier_uniform_
import crash_on_ipy
import copy


def select_task(opt):
    if opt.task == 'repeat':
        return repeat
    elif opt.task == 'mirror':
        return mirror
    elif opt.task == 'm10ae':
        return m10ae
    elif opt.task == 'sort':
        return sort
    elif opt.task == 'repeat_mky':
        return repeat_mky
    elif opt.task == 'mirror_mky':
        return mirror_mky
    else:
        raise ModuleNotFoundError


if __name__ == '__main__':

    opt = utils.parse_opts('main.py')
    utils.init_seed(opt.seed)
    utils.param_str(opt)
    task = select_task(opt)
    task.train(opt)
