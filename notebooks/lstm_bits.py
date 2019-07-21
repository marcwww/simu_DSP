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


def gen_seq(bsz):
    seq = torch.randint(0, 6, (bsz, 4))
    return embeddings(seq)


def gen_batch(bsz=1):
    idim = 6 + 1
    width = idim - 1

    seq_len = 4
    seq = gen_seq(bsz)
    seq = torch.Tensor(seq)
    inp = torch.zeros(bsz, seq_len + 1, width + 1)
    inp[:, :seq_len, :width] = seq
    inp[:, seq_len, width] = 1.0  # delimiter in our control channel

    outp = seq.clone()

    return inp.float(), outp.float()


class LSTM(nn.Module):

    def __init__(self, idim, odim, hdim):
        super(LSTM, self).__init__()
        self.idim = idim
        self.odim = odim
        self.hdim = hdim
        self.w_ih = nn.Parameter(torch.Tensor(4 * hdim, idim))
        self.w_hh = nn.Parameter(torch.Tensor(4 * hdim, hdim))
        self.b = nn.Parameter(torch.Tensor(4 * hdim))
        self.clf = nn.Sequential(nn.Linear(hdim, odim), nn.Sigmoid())
        self.reset_params()

    def reset_params(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                xavier_uniform_(p)

    def trans(self, inp, h, c):
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
        h_new = o * torch.tanh(c_new)
        return h_new, c_new, i, f, g, o

    def hidden_init(self, bsz):
        return torch.zeros(bsz, self.hdim)

    def forward(self, inp, tlen, retain_grad=False):
        # inp: (bsz, ilen, idim)
        bsz, ilen, _ = inp.shape
        self.inp = inp
        inp_padded = F.pad(inp, [0, 0, 0, tlen], 'constant', 0)
        # inp_padded: (bsz, ilen + tlen, idim)
        h = self.hidden_init(bsz)
        c = self.hidden_init(bsz)
        hs = []
        cs = []
        igates = []
        fgates = []
        ggates = []
        ogates = []
        for i in range(ilen + tlen):
            h, c, i, f, g, o = self.trans(inp_padded[:, i], h, c)
            hs.append(h)
            cs.append(c)
            igates.append(i)
            fgates.append(f)
            ggates.append(g)
            ogates.append(o)
        self.hs = torch.stack(hs, dim=1)  # outp: (bsz, seq_len, hdim)
        self.cs = torch.stack(cs, dim=1)
        self.igates = torch.stack(igates, dim=1)
        self.fgates = torch.stack(fgates, dim=1)
        self.ggates = torch.stack(ggates, dim=1)
        self.ogates = torch.stack(ogates, dim=1)

        probs = self.clf(self.hs)  # probs: (bsz, seq_len, odim)

        if retain_grad:
            self.retain_grad()
        return probs[:, ilen:]

    def retain_grad(self):
        self.inp.retain_grad()
        self.hs.retain_grad()
        self.cs.retain_grad()
        self.igates.retain_grad()
        self.fgates.retain_grad()
        self.ggates.retain_grad()
        self.ogates.retain_grad()

def run_iter(model, batch, criterion, optimizer, is_training):
    model.train(is_training)
    (inp, tar) = batch
    tlen = tar.shape[1]
    if is_training:
        out = model(inp, tlen)
        loss = criterion(out, tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
    else:
        out = model(inp, tlen)
        pred = out.data.gt(0.5).float()
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
            out_binarized = run_iter(model, batch, None, None, is_training=False)
            # out_binarized: (bsz, tlen=4, odim=6)
            nc += torch.abs(out_binarized - tar).sum(dim=-1).eq(0).sum(dim=0)
            nt += bsz

    acc_along = (nc.float() / nt).cpu().numpy().tolist()
    acc_along = [round(acc, 4) for acc in acc_along]
    acc_mean = np.mean(acc_along)

    return acc_along, acc_mean


def train(model, nseqs_train, nseqs_valid, valid_times=10):
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-3)
    criterion = nn.BCELoss()
    losses = []
    logs = []
    valid_per = int(nseqs_train / valid_times)
    for n in range(nseqs_train):
        batch = gen_batch()
        loss = run_iter(model, batch, criterion, optimizer, is_training=True)
        losses.append(loss.item())
        if (n + 1) % valid_per == 0:
            acc_along, acc_mean = valid(model, nseqs_valid)
            log = {'#seqs': n,
                   'acc': round(acc_mean, 4),
                   'acc_along': [round(acc, 4) for acc in acc_along],
                   'loss': round(np.mean(losses), 4)}
            logs.append(log)
            losses = []
            print(log)
    return logs


def init_seed(seed=100):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    # lstm = LSTM(6 + 1, 6, 100)
    # inp = torch.Tensor(1, 4, 6 + 1)
    # outp = lstm(inp, 4)
    init_seed()
    hdim = 100
    mdl = LSTM(6 + 1, 6, hdim)
    train(mdl, 20000, 200, 100)
