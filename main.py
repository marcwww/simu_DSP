import nets
from macros import *
import torch
import utils
from hparams import opts
import argparse
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tasks import repeat
from torch.nn.init import orthogonal_, uniform_, xavier_uniform_
import sys
import crash_on_ipy
import copy

def select_task(opt):
    gen_batch = None
    train = None
    Model = None
    diter_train = None
    diter_valid = None

    if opt.task == 'repeat':
        gen_batch = repeat.gen_batch
        train = repeat.train
        Model = repeat.Model
        diter_train = utils.DataIter(opt, opt.nbatch_train, gen_batch)
        opt_test = copy.deepcopy(opt)
        opt_test.min_len = 6
        opt_test.max_len = 10
        diter_valid = utils.DataIter(opt_test, opt.nbatch_valid, gen_batch)

    return gen_batch, train, Model, diter_train, diter_valid

def select_encoder(opt):
    encoder = None
    if opt.enc_type == 'lstm':
        encoder = nets.EncoderLSTM(opt.idim, opt.hdim, opt.dropout)
    elif opt.enc_type == 'organics':
        encoder = nets.EncoderORGaNICs(opt.idim, opt.hdim, opt.dropout)
    elif opt.enc_type == 'ntm':
        encoder = nets.EncoderNTM(opt.idim, opt.hdim, opt.N, opt.M, opt.dropout, opt.read_first)
    else:
        raise ModuleNotFoundError

    return encoder

def select_optim(opt, model):
    optimizer = None

    if opt.optim == 'rmsprop':
        optimizer = optim.RMSprop(params=filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=opt.lr)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=opt.lr)
    else:
        raise ModuleNotFoundError

    return optimizer

if __name__ == '__main__':

    opt = utils.parse_opts('main.py')
    utils.init_seed(opt.seed)

    gen_batch, train, Model, diter_train, diter_valid = select_task(opt)
    encoder = select_encoder(opt)
    model = Model(encoder, opt.odim)
    utils.init_model(model, xavier_uniform_)

    if opt.fload is not None and opt.continue_training:
        utils.model_loading(opt, model)

    optimizer = select_optim(opt, model)
    param_str = utils.param_str(opt)
    for key, val in param_str.items():
        print(str(key) + ': ' + str(val))

    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.1,
                                  patience=opt.patience,
                                  min_lr=opt.lr/10)

    train(opt=opt,
          model=model,
          diter_train=diter_train,
          diter_valid=diter_valid,
          optim=optimizer,
          scheduler=scheduler)
