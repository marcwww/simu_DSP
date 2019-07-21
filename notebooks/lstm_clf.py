import torchtext
from torchtext.data import Dataset
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import os
from torch.nn import functional as F
from torch import optim
import json
import tqdm
import logging
from collections import defaultdict
from torch.nn.init import xavier_uniform_

# import crash_on_ipy

embeddings = nn.Embedding(6, 6)
embeddings.weight = nn.Parameter(torch.eye(6), requires_grad=False)


def build_batch(x0, x1, x2, x3):
    seq = torch.LongTensor([[x0, x1, x2, x3]])
    seq = embeddings(seq)
    idim = 6 + 1
    width = idim - 1
    seq_len = 4
    inp = torch.zeros(1, seq_len + 1, width + 1)
    inp[:, :seq_len, :width] = seq
    inp[:, seq_len, width] = 1.0  # delimiter in our control channel
    return inp.float()


inp_lst = []
name_lst = []
for x0 in range(6):
    for x1 in range(6):
        for x2 in range(6):
            for x3 in range(6):
                inp = build_batch(x0, x1, x2, x3)
                inp_lst.append(inp)
                name_lst.append(f'{x0}{x1}{x2}{x3}')


def gen_seq(bsz):
    seq = torch.randint(0, 6, (bsz, 4))
    return seq, embeddings(seq)


def gen_batch(bsz=1):
    idim = 6 + 1
    width = idim - 1

    seq_len = 4
    seq, inp_ctnt = gen_seq(bsz)
    inp_ctnt = torch.Tensor(inp_ctnt)
    inp = torch.zeros(bsz, seq_len + 1, width + 1)
    inp[:, :seq_len, :width] = inp_ctnt
    inp[:, seq_len, width] = 1.0  # delimiter in our control channel

    tar = seq

    return inp.float(), tar


class LSTM(nn.Module):

    def __init__(self, idim, nclasses, hdim):
        super(LSTM, self).__init__()
        self.idim = idim
        self.nclasses = nclasses
        self.hdim = hdim
        self.w_ih = nn.Parameter(torch.Tensor(4 * hdim, idim))
        self.w_hh = nn.Parameter(torch.Tensor(4 * hdim, hdim))
        self.b = nn.Parameter(torch.Tensor(4 * hdim))
        # self.clf = nn.Sequential(nn.Linear(hdim, nclasses), nn.Softmax(dim=-1))
        self.clf = nn.Sequential(nn.Linear(hdim, nclasses))
        self.reset_params()

    def reset_params(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                xavier_uniform_(p)

    def trans(self, inp, h, c, retain_grad=False, ablate_cindices=None):
        # inp: (bsz, idim)
        w = torch.cat([self.w_ih, self.w_hh], dim=-1)  # w: (4 * hdim, idim + hdim)
        x = torch.cat([inp, h], dim=-1)  # x: (bsz, idim + hdim)
        b = self.b  # b: (4 * hdim, )
        out_linear = w[None].matmul(x[:, :, None]).squeeze(-1) + b  # out_linear: (bsz, 4 * hdim)
        i, f, g, o = out_linear.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        if ablate_cindices is not None:
            # c_new: (bsz, hdim)
            c_new[:, ablate_cindices] = 0

        h_new = o * torch.tanh(c_new)
        if retain_grad:
            i.retain_grad()
            f.retain_grad()
            g.retain_grad()
            o.retain_grad()
            c_new.retain_grad()
            h_new.retain_grad()
        return h_new, c_new, i, f, g, o

    def hidden_init(self, bsz):
        return torch.zeros(bsz, self.hdim)

    def get_grad(self, hs):
        if hs in ['inp']:
            return getattr(self, hs).grad
        else:
            hs = getattr(self, hs)
            grad = torch.stack([h.grad for h in hs], dim=1)
            return grad

    def get_hs(self, hs):
        if hs in ['inp']:
            return getattr(self, hs)
        else:
            hs = getattr(self, hs)
            hs = torch.stack([h for h in hs], dim=1)
            return hs

    def forward(self, inp, tlen, retain_grad=False, ablate_cindices=None):
        # inp: (bsz, ilen, idim)
        bsz, ilen, _ = inp.shape
        self.inp = inp
        inp_padded = F.pad(inp, [0, 0, 0, tlen], 'constant', 0)
        # inp_padded: (bsz, ilen + tlen, idim)
        h = self.hidden_init(bsz)
        c = self.hidden_init(bsz)
        self.hs = []
        self.cs = []
        self.igates = []
        self.fgates = []
        self.ggates = []
        self.ogates = []
        for i in range(ilen + tlen):
            h, c, i, f, g, o = self.trans(inp_padded[:, i], h, c,
                                          retain_grad=retain_grad,
                                          ablate_cindices=ablate_cindices)
            self.hs.append(h)
            self.cs.append(c)
            self.igates.append(i)
            self.fgates.append(f)
            self.ggates.append(g)
            self.ogates.append(o)
        # self.hs = torch.stack(self.hs, dim=1)  # outp: (bsz, seq_len, hdim)
        hs = torch.stack(self.hs, dim=1)
        probs = self.clf(hs)  # probs: (bsz, seq_len, nclasses)

        if retain_grad:
            self.inp.retain_grad()

        return probs[:, ilen:]


def run_iter(model, batch, criterion, optimizer, is_training,
             ablate_cindices=None):
    model.train(is_training)
    (inp, tar) = batch
    tlen = tar.shape[1]
    if is_training:
        out = model(inp, tlen)
        bsz, _, nclasses = out.shape
        # out: (bsz, tlen, nclasses)
        # tar: (bsz, tlen)
        loss = criterion(out.view(-1, nclasses), tar.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
    else:
        out = model(inp, tlen, ablate_cindices=ablate_cindices)
        pred = out.max(dim=-1)[1]
        # pred: (bsz, tlen)
        return pred


def valid(model, nseqs):
    nc = 0
    nt = 0
    with torch.no_grad():
        model.eval()
        for n in range(nseqs):
            batch = gen_batch()
            (inp, tar) = batch
            bsz = inp.shape[0]
            pred = run_iter(model, batch, None, None, is_training=False)
            nc += (pred == tar).sum(dim=0)  # nc: (tlen,)
            nt += bsz

    acc_along = (nc.float() / nt).cpu().numpy().tolist()
    acc_along = [round(acc, 4) for acc in acc_along]
    acc_mean = np.mean(acc_along)

    return acc_along, acc_mean


def ablation_test(model, nseqs, ablate_cindices):
    nc = 0
    nt = 0
    with torch.no_grad():
        model.eval()
        for n in range(nseqs):
            batch = gen_batch()
            (inp, tar) = batch
            bsz = inp.shape[0]
            pred = run_iter(model, batch, None, None, is_training=False,
                            ablate_cindices=ablate_cindices)
            nc += (pred == tar).sum(dim=0)  # nc: (tlen,)
            nt += bsz

    acc_along = (nc.float() / nt).cpu().numpy().tolist()
    acc_along = [round(acc, 4) for acc in acc_along]
    acc_mean = np.mean(acc_along)

    return acc_along, acc_mean


def train(model, nseqs_train, nseqs_valid, valid_times=10):
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    losses = []
    logs = []
    valid_per = int(nseqs_train / valid_times)
    remain_set = set(name_lst)
    total = len(name_lst)

    for n in range(nseqs_train):
        batch = gen_batch()
        seq = batch[1]  # seq: (bsz, ilen)
        seq = ''.join(list(map(str, seq[0].numpy().tolist())))
        if seq in remain_set:
            remain_set.remove(seq)

        loss = run_iter(model, batch, criterion, optimizer, is_training=True)
        losses.append(loss.item())
        if (n + 1) % valid_per == 0:
            acc_along, acc_mean = valid(model, nseqs_valid)
            log = {'#seqs': n,
                   'acc': round(acc_mean, 4),
                   'acc_along': [round(acc, 4) for acc in acc_along],
                   'loss': round(np.mean(losses), 4),
                   'seen%': round(1 - len(remain_set) / total, 2)}
            logs.append(log)
            losses = []
            print(log)
    return logs


def init_seed(seed=100):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    # mdl = torch.load('lstm-clf-repeat-mky-whole.mdl')
    # gate_l_lst = []
    # gate_r_lst = []
    # for x0 in range(6):
    #     for x1 in range(6):
    #         for x2 in range(6):
    #             for x3 in range(6):
    #                 inp = build_batch(x0, x1, x2, x3)
    #                 tar = mdl(inp, 4)
    #                 #                     print(f'{x0}{x1}{x2}{x3}')
    #                 gate = mdl.get_hs('igates')
    #                 gate = mdl.get_hs('igates')  # (bsz=1, ilen + tlen, hdim)
    #                 gate_l = gate.le(0.1)
    #                 gate_r = gate.ge(0.9)
    #                 gate_l_lst.append(gate_l)
    #                 gate_r_lst.append(gate_r)
    # gate_l_lst = torch.cat(gate_l_lst, dim=0)  # (nseqs, ilen + tlen, hdim)
    # gate_r_lst = torch.cat(gate_r_lst, dim=0)  # (nseqs, ilen + tlen, hdim)
    # ratio_l = gate_l_lst.sum(dim=-1).float() / gate_l_lst.shape[-1]  # (nseqs, ilen + tlen)
    # ratio_r = gate_r_lst.sum(dim=-1).float() / gate_r_lst.shape[-1]
    # ratio_l = ratio_l.detach().numpy()
    # ratio_r = ratio_r.detach().numpy()
    # print(ratio_l)

    # lstm = LSTM(6 + 1, 6, 100)
    # inp = torch.Tensor(1, 4, 6 + 1)
    # outp = lstm(inp, 4)
    # init_seed()
    hdim = 100
    mdl = LSTM(6 + 1, 6, hdim)
    train(mdl, 20000, 200, 100)
