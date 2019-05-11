import torchtext
from macros import *
from torchtext.data import Dataset
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import utils
import random
import os
from torch.nn import functional as F
import json
import nets
import tqdm
import logging
from .m10ae_utils import gen_batch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')
import copy
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score


def gen_batch_train(opt):
    return gen_batch(opt.min_len_train, opt.max_len_train, opt.min_lopr_train, opt.max_lopr_train, opt.modd, opt.bsz)


def gen_batch_valid(opt):
    return gen_batch(opt.min_len_valid, opt.max_len_valid, opt.min_lopr_valid, opt.max_lopr_valid, opt.modd, opt.bsz)


def log_init(opt):
    basename = "{}-{}-{}-{}".format(opt.task,
                                    opt.sub_task,
                                    opt.enc_type,
                                    utils.time_int())
    log_fname = basename + ".json"
    log_path = os.path.join(LOGS, log_fname)
    with open(log_path, 'w') as f:
        f.write(str(utils.param_str(opt)) + '\n')
    logging.info(f'Logging file path: {log_path}')
    return log_path, basename


def log_print(log_path, log_str, optim):
    log_str = json.dumps(log_str)
    logging.info(f'{log_str}')
    with open(log_path, 'a+') as f:
        f.write(log_str + '\n')

    # for param_group in optim.param_groups:
    #     print('learning rate:', param_group['lr'])


def valid_along(**kwargs):
    model = kwargs['model']
    diter_valid = kwargs['diter_valid_along']

    nc = 0
    nt = 0

    with torch.no_grad():
        model.eval()
        for inp, tar in diter_valid:
            tlen, bsz, _ = tar.shape
            # out: (seq_len, bsz, odim)
            out = model(inp, tlen)
            pred = out.max(dim=-1)[1]
            nc += torch.abs(out_binarized - tar).sum(dim=-1).eq(0).sum(dim=1)
            nt += bsz

    return (nc.float() / nt).cpu().numpy().tolist()


def valid(model, valid_iter, args):
    pred_lst = []
    tar_lst = []

    with torch.no_grad():
        model.eval()
        for batch in valid_iter:
            inp, tar = batch
            out = run_iter(model, batch, None, None, args, is_training=False)
            pred = out.max(dim=-1)[1]
            pred_lst.extend(pred.cpu().numpy())
            tar_lst.extend(tar.cpu().numpy())

    return accuracy_score(tar_lst, pred_lst)


def run_iter(model, batch, criterion, optimizer, args, is_training):
    model.train(is_training)
    (inp, tar) = batch
    if is_training:
        out = model(inp)
        loss = criterion(out, tar)
        optimizer.zero_grad()
        loss.backward()
        gnorm = clip_grad_norm_(parameters=model.parameters(), max_norm=args.gclip)
        optimizer.step()
        return loss, gnorm
    else:
        out = model(inp)
        pred = out.data.gt(0.5).float()
        return pred


def train(args):
    encoder = nets.select_enc(args)
    device = utils.build_device(args)
    model = Model(encoder, args).to(device)
    utils.init_model(model)
    if args.fload is not None and args.continue_training:
        utils.model_loading(args, model)
    train_iter = utils.DataIter(args, args.nbatch_train, gen_batch_train)
    valid_iter = utils.DataIter(args, args.nbatch_valid, gen_batch_valid)
    optim = utils.select_optim(args, model)
    scheduler = ReduceLROnPlateau(optim,
                                  mode='min',
                                  factor=0.1,
                                  patience=args.patience,
                                  min_lr=args.lr / 10)

    criterion = nn.CrossEntropyLoss()
    log_path, basename = log_init(args)

    best_perform = -1
    losses = []
    gnorms = []
    for epoch in range(args.nepoch):
        train_iter_tqdm = tqdm.tqdm(train_iter)
        for i, batch in enumerate(train_iter_tqdm):
            loss, gnorm = run_iter(model, batch, criterion, optim, args, is_training=True)
            loss = loss.item()
            losses.append(loss)
            gnorms.append(gnorm)
            train_iter_tqdm.set_description(f'Epoch {epoch} loss {loss:.4f}')

            if (i + 1) % (len(train_iter_tqdm) // args.valid_times) == 0:
                loss_ave = np.mean(losses)
                gnorm_ave = np.mean(gnorms)
                losses = []
                gnorms = []
                acc = valid(model, valid_iter, args)
                log_str = {'Epoch': epoch,
                           'acc': round(acc, 4),
                           'loss': round(float(loss_ave), 4),
                           'gnorm': round(float(gnorm_ave), 4)}

                log_print(log_path, log_str, optim)
                scheduler.step(loss_ave)

                if acc >= best_perform:
                    best_perform = acc
                    utils.mdl_save(model, basename, epoch, loss_ave, acc)


class Model(nn.Module):

    def __init__(self, encoder, args):
        super(Model, self).__init__()
        odim = args.odim
        self.encoder = encoder
        self.hdim = self.encoder.odim
        self.odim = odim
        self.clf = nn.Sequential(nn.Linear(self.hdim, self.odim))

    def forward(self, inp):
        out = self.encoder(embs=inp)
        encoded = out[-1]
        logtis = self.clf(encoded)

        return logtis
