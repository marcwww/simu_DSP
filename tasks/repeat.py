import torchtext
from macros import *
from torchtext.data import Dataset
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import utils
import random
import os
from torch.nn import functional as F
import json
import copy
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score

def gen_batch(opt):

    min_len = opt.min_len
    max_len = opt.max_len
    bsz = opt.bsz
    width = opt.idim
    delimiter = opt.delimiter

    seq_len = random.randint(min_len, max_len)
    if delimiter:
        assert width > 1

        seq = np.random.binomial(1, 0.5, (seq_len, bsz, width - 1))
        seq = torch.Tensor(seq)
        inp = torch.zeros(seq_len + 1, bsz, width)
        inp[:seq_len, :, :width-1] = seq
        inp[seq_len, :, width-1] = 1.0  # delimiter in our control channel

    else:
        seq = np.random.binomial(1, 0.5, (seq_len, bsz, width))
        seq = torch.Tensor(seq)
        inp = seq

    outp = seq.clone()

    return inp.float(), outp.float()

def log_init(opt):
    basename = "{}-{}-{}-{}".format(opt.task,
                                opt.sub_task,
                                opt.enc_type,
                                utils.time_int())
    log_fname = basename + ".json"
    log_path = os.path.join(LOGS, log_fname)
    with open(log_path, 'w') as f:
        f.write(str(utils.param_str(opt)) + '\n')

    return log_path, basename

def log_print(log_path, epoch, acc, loss_ave, optim):
    log_str = {'Epoch': epoch,
               'Format':'a/l',
               'Metrics': [round(acc, 4), round(loss_ave, 4)]}
    log_str = json.dumps(log_str)
    print(log_str)
    with open(log_path, 'a+') as f:
        f.write(log_str + '\n')

    for param_group in optim.param_groups:
        print('learning rate:', param_group['lr'])

def mdl_save(model, basename):
    model_fname = basename + ".model"
    save_path = os.path.join(MDLS, model_fname)
    print('Saving to ' + save_path)
    torch.save(model.state_dict(), save_path)

def valid(**kwargs):
    model = kwargs['model']
    diter_valid = kwargs['diter_valid']

    nc = 0
    nt = 0

    with torch.no_grad():
        model.eval()
        for inp, tar in diter_valid:
            tlen, bsz, _ = tar.shape
            out = model(inp, tlen)
            out_binarized = out.data.gt(0.5).float()

            cost = torch.abs(out_binarized-tar).sum(dim=0).sum(dim=-1)
            nc += cost.eq(0).sum()
            nt += bsz

    return nc.item()/nt

def train(**kwargs):
    opt = kwargs['opt']
    model = kwargs['model']
    diter_train = kwargs['diter_train']
    diter_valid = kwargs['diter_valid']
    optim = kwargs['optim']
    scheduler = kwargs['scheduler']

    criterion = nn.BCELoss()
    log_path, basename = log_init(opt)

    best_perform = 0
    losses = []
    for epoch in range(opt.nepoch):
        for i, (inp, tar) in enumerate(diter_train):
            model.train()
            model.zero_grad()
            tlen = tar.shape[0]

            out = model(inp, tlen)
            loss = criterion(out, tar)
            losses.append(loss.item())

            loss.backward()
            clip_grad_norm_(model.parameters(), opt.gclip)
            optim.step()

            loss = {'loss': loss.item()}
            percent = i / len(diter_train)
            utils.progress_bar(percent, loss, epoch)

            if (i + 1) % (len(diter_train) // opt.valid_times) == 0:
                loss_ave = np.mean(losses)
                losses = []
                acc = valid(model=model, diter_valid=diter_valid)

                log_print(log_path, epoch, acc, loss_ave, optim)
                scheduler.step(loss_ave)

                if acc > best_perform:
                    best_perform = acc
                    mdl_save(model, basename)

class Model(nn.Module):

    def __init__(self, encoder, odim):
        super(Model, self).__init__()
        self.encoder = encoder
        self.hdim = self.encoder.odim
        self.odim = odim
        self.clf = nn.Sequential(nn.Linear(self.hdim, self.odim),
                                 nn.Sigmoid())

    def forward(self, inp, tlen):
        ilen = inp.shape[0]
        inp_padded = F.pad(inp, (0, 0, 0, 0, 0, tlen), 'constant', 0)
        out = self.encoder(embs=inp_padded, ilen=ilen)
        probs = self.clf(out)

        return probs[ilen:]











