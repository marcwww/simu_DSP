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
    analy = None
    Model = None
    diter_analy = None

    if opt.task == 'repeat':
        gen_batch_analy = repeat.gen_batch_analy
        analy = repeat.analy
        Model = repeat.Model
        diter_analy = utils.DataIter(opt, 500, gen_batch_analy)

    return gen_batch, analy, Model, diter_analy

def select_encoder(opt):

    if opt.enc_type == 'lstm':
        encoder = nets.EncoderLSTM(opt.idim, opt.hdim, opt.dropout)
    elif opt.enc_type == 'organics':
        encoder = nets.EncoderORGaNICs(opt.idim, opt.hdim, opt.dropout)
    elif opt.enc_type == 'ntm':
        encoder = nets.EncoderNTM(opt.idim, opt.hdim, opt.N, opt.M, opt.dropout, opt.read_first)
    elif opt.enc_type == 'ntmnos':
        encoder = nets.EncoderNTMnos(opt.idim, opt.hdim, opt.N, opt.M, opt.dropout, opt.read_first)
    elif opt.enc_type == 'ntmr':
        encoder = nets.EncoderNTMr(opt.idim, opt.hdim, opt.N, opt.M, opt.dropout, opt.read_first)
    else:
        raise ModuleNotFoundError

    return encoder

def select_optim(opt, model):

    if opt.optim == 'rmsprop':
        optimizer = optim.RMSprop(params=filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=opt.lr)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=opt.lr)
    else:
        raise ModuleNotFoundError

    return optimizer

def model_loading(opt, model):
    model_fname = opt.fload
    location = {'cuda:' + str(opt.gpu): 'cuda:' + str(opt.gpu)} if opt.gpu != -1 else 'cpu'
    model_path = os.path.join(MDLS, model_fname)
    model_path = os.path.join('..', model_path)
    model_dict = torch.load(model_path, map_location=location)
    model.load_state_dict(model_dict)
    print('Loaded from ' + model_path)

if __name__ == '__main__':

    opt = utils.parse_opts('main.py')

    assert opt.enc_type == 'ntmr'
    assert opt.task == 'repeat'

    utils.init_seed(opt.seed)

    gen_batch, analy, Model, diter_analy = select_task(opt)
    encoder = select_encoder(opt)
    model = Model(encoder, opt.odim)
    utils.init_model(model, xavier_uniform_)
    model_loading(opt, model)

    optimizer = select_optim(opt, model)
    param_str = utils.param_str(opt)
    for key, val in param_str.items():
        print(str(key) + ': ' + str(val))

    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.1,
                                  patience=opt.patience,
                                  min_lr=opt.lr/10)

    fname_dict = {}
    fanaly_name = 'memcontent'

    fenc = os.path.join('..', os.path.join(ANAs, '%s-%s-%s-%d.txt' % (opt.task, fanaly_name, opt.enc_type, utils.time_int())))
    fname_dict['f' + opt.enc_type] = fenc
    flstm = os.path.join('..', os.path.join(ANAs, '%s-%s-%s_lstm-%d.txt' % (opt.task, fanaly_name, opt.enc_type, utils.time_int())))
    fname_dict['flstm'] = flstm

    with utils.analy(model.encoder, fname_dict):
        acc = analy(model=model, diter_analy=diter_analy, enc_type=opt.enc_type)

    print('Accuracy', acc)
    print('Saved to', fname_dict)