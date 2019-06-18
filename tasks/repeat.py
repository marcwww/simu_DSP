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
from collections import defaultdict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')
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

    outp = seq.clone()

    return inp.float(), outp.float()


def gen_batch_train(opt):
    return gen_batch(opt.min_len_train, opt.max_len_train, opt.bsz, opt.idim)


def gen_batch_valid(opt):
    return gen_batch(opt.min_len_valid, opt.max_len_valid, opt.bsz, opt.idim)


def gen_batch_test(opt):
    return gen_batch(opt.min_len_train, opt.max_len_valid, opt.bsz, opt.idim)


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

    outp = seq.clone()

    return inp.float(), outp.float()


def gen_batch_pattern(opt):
    NUMS = list(range(0, 6))
    seq_len = 4
    bsz = 1
    idim = opt.idim
    width = idim - 1

    seq = []
    numeral = random.choice(NUMS)
    pattern = opt.pattern
    assert len(pattern) == 3
    pattern = list(map(int, pattern))
    delta = pattern + [-1]

    for i in range(4):
        bivec = utils.bin_vec(numeral, width)
        seq.append(bivec)
        numeral += delta[i]
        numeral %= 6

    seq = torch.Tensor(seq).unsqueeze(1)
    inp = torch.zeros(seq_len + 1, bsz, width + 1)
    inp[:seq_len, :, :width] = seq
    inp[seq_len, :, width] = 1.0  # delimiter in our control channel

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
    logging.info(f'Logging file path: {log_path}')
    return log_path, basename


def log_print(log_path, log_str, optim):
    log_str = json.dumps(log_str)
    logging.info(f'{log_str}')
    with open(log_path, 'a+') as f:
        f.write(log_str + '\n')

    # for param_group in optim.param_groups:
    #     print('learning rate:', param_group['lr'])


def analy(**kwargs):
    model = kwargs['model']
    diter_analy = kwargs['diter_analy']
    enc_type = kwargs['enc_type']
    fanalysis = getattr(model.encoder, 'f' + enc_type)

    nc = 0
    nt = 0

    with torch.no_grad():
        model.eval()
        for i, (inp, tar) in enumerate(tqdm.tqdm(diter_analy)):
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
            # print(line)
            print(line, file=fanalysis)

    return nc / nt


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
            out_binarized = out.data.gt(0.5).float()

            nc += torch.abs(out_binarized - tar).sum(dim=-1).eq(0).sum(dim=1)
            nt += bsz

    return (nc.float() / nt).cpu().numpy().tolist()


def test(args):
    encoder = nets.select_enc(args)
    model = Model(encoder, args)
    utils.init_model(model)
    utils.model_loading(args, model, True)
    test_iter = utils.DataIter(args, args.nbatch_test, gen_batch_test)
    nc = defaultdict(int)
    nt = defaultdict(int)
    acc = defaultdict(float)
    with torch.no_grad():
        model.eval()
        for inp, tar in tqdm.tqdm(test_iter):
            tlen, bsz, _ = tar.shape
            # out: (seq_len, bsz, odim)
            out = model(inp, tlen)
            out_binarized = out.data.gt(0.5).float()

            corrects = torch.abs(out_binarized - tar).sum(dim=0).sum(dim=-1).eq(0)
            for b in range(bsz):
                nc[tlen] += corrects[b].item()
                nt[tlen] += 1

    for key in nc.keys():
        acc[key] = nc[key]/nt[key]

    return sorted(acc.items()), sorted(nt.items())


def valid(model, valid_iter, args):
    nc = 0
    nt = 0

    with torch.no_grad():
        model.eval()
        for batch in valid_iter:
            inp, tar = batch
            tlen, bsz, _ = tar.shape
            out_binarized = run_iter(model, batch, None, None, args, is_training=False)
            cost = torch.abs(out_binarized - tar).sum(dim=0).sum(dim=-1)
            nc += cost.eq(0).sum()
            nt += bsz

    return nc.item() / nt


def run_iter(model, batch, criterion, optimizer, args, is_training):
    model.train(is_training)
    (inp, tar) = batch
    tlen = tar.shape[0]
    if is_training:
        out = model(inp, tlen)
        loss = criterion(out, tar)
        optimizer.zero_grad()
        loss.backward()
        gnorm = clip_grad_norm_(parameters=model.parameters(), max_norm=args.gclip)
        optimizer.step()
        return loss, gnorm
    else:
        out = model(inp, tlen)
        pred = out.data.gt(0.5).float()
        return pred


def train(args):
    encoder = nets.select_enc(args)
    model = Model(encoder, args)
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

    criterion = nn.BCELoss()
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
        self.clf = nn.Sequential(nn.Linear(self.hdim, self.odim),
                                 nn.Sigmoid())

    def forward(self, inp, tlen):
        ilen = inp.shape[0]
        inp_padded = F.pad(inp, [0, 0, 0, 0, 0, tlen], 'constant', 0)
        out = self.encoder(embs=inp_padded, ilen=ilen)
        probs = self.clf(out)

        return probs[ilen:]
