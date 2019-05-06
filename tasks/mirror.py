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


def gen_batch(min_len, max_len, bsz, idim):
    min_len = min_len
    max_len = max_len
    bsz = bsz
    assert idim > 2
    width = idim - 1

    seq_len = random.randint(min_len, max_len)

    seq = np.random.binomial(1, 0.5, (seq_len, bsz, width))
    seq = torch.Tensor(seq)
    inp = torch.zeros(seq_len + 1, bsz, width + 1)
    inp[:seq_len, :, :width] = seq
    inp[seq_len, :, width] = 1.0  # delimiter in our control channel

    outp = seq[range(seq_len - 1, -1, -1)].clone()

    return inp.float(), outp.float()


def gen_batch_train(opt):
    return gen_batch(opt.min_len_train, opt.max_len_train, opt.bsz, opt.idim)


def gen_batch_valid(opt):
    return gen_batch(opt.min_len_valid, opt.max_len_valid, opt.bsz, opt.idim)


def gen_batch_analy(opt):
    seq_len = 4
    bsz = 1
    idim = opt.idim
    assert idim > 2
    width = idim - 1

    NUMS = list(range(1, 9 + 1))

    seq = []
    for _ in range(seq_len):
        numeral = random.choice(NUMS)
        bivec = utils.bin_vec(numeral, width)
        seq.append(bivec)
    seq = torch.Tensor(seq).unsqueeze(1)
    inp = torch.zeros(seq_len + 1, bsz, width + 1)
    inp[:seq_len, :, :width] = seq
    inp[seq_len, :, width] = 1.0  # delimiter in our control channel

    outp = seq[range(seq_len - 1, -1, -1)].clone()

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
               'Format': 'a/l',
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


def analy(**kwargs):
    model = kwargs['model']
    diter_analy = kwargs['diter_analy']
    enc_type = kwargs['enc_type']
    fanalysis = getattr(model.encoder, 'f' + enc_type)

    nc = 0
    nt = 0

    with torch.no_grad():
        model.eval()
        for i, (inp, tar) in enumerate(diter_analy):
            tlen, bsz, _ = tar.shape
            assert bsz == 1
            out = model(inp, tlen)
            out_binarized = out.data.gt(0.5).float()

            cost = torch.abs(out_binarized - tar).sum(dim=0).sum(dim=-1)
            nc += cost.eq(0).sum().item()
            nt += bsz

            is_correct = 1 if (cost == 0) else 0
            seq_inp = [str(utils.bivec_tensor2int(num_vec)) for num_vec in inp[:, 0, :-1]]
            seq_tar = [str(utils.bivec_tensor2int(num_vec)) for num_vec in tar[:, 0]]
            line = {'type': 'input',
                    'idx': i,
                    'inp': seq_inp,
                    'tar': seq_tar,
                    'is_correct': is_correct}
            line = json.dumps(line)
            print(line)
            print(line, file=fanalysis)

    return nc / nt


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

            cost = torch.abs(out_binarized - tar).sum(dim=0).sum(dim=-1)
            nc += cost.eq(0).sum()
            nt += bsz

    return nc.item() / nt


def train(**kwargs):
    opt = kwargs['opt']
    model = kwargs['model']
    diter_train = kwargs['diter_train']
    diter_valid = kwargs['diter_valid']
    optim = kwargs['optim']
    scheduler = kwargs['scheduler']

    criterion = nn.BCELoss()
    log_path, basename = log_init(opt)

    best_perform = -1
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

                if acc >= best_perform:
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
