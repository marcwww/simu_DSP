import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
import json
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack
from .MANN import MANNBaseEncoder
import time


class NTMMemory(nn.Module):

    def __init__(self, N, M):
        super(NTMMemory, self).__init__()
        self.N = N
        self.M = M
        self.mem_bias = nn.Parameter(torch.Tensor(N, M),
                                     requires_grad=False)
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform_(self.mem_bias, -stdev, stdev)

    def reset(self, bsz):
        self.bsz = bsz
        self.memory = self.mem_bias.expand(bsz, self.N, self.M)

    def size(self):
        return self.N, self.M

    def read(self, w):
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        self.pre_mem = self.memory
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.pre_mem * (1 - erase) + add

    def _similarity(self, k, beta):
        k = k.view(self.bsz, 1, -1)
        w = F.softmax(beta *
                      F.cosine_similarity(self.memory + 1e-16,
                                          k + 1e-16, dim=-1),
                      dim=1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        res = utils.modulo_convolve(wg, s)
        return res

    def _sharpen(self, ww, gamma):
        w = ww ** gamma
        w = torch.div(w,
                      torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w

    def _preproc(self, k, beta, g, s, gamma):
        k = k.clone()
        beta = F.softplus(beta)
        g = torch.sigmoid(g)
        s = F.softmax(s, dim=1)
        gamma = 1 + F.softplus(gamma)

        return k, beta, g, s, gamma

    def address(self, k, beta, g, s, gamma, w_pre):
        t = time.time()
        k, beta, g, s, gamma = \
            self._preproc(k, beta, g, s, gamma)
        # print('_preproc', time.time() - t)
        # t = time.time()
        wc = self._similarity(k, beta)
        # print('_similarity', time.time() - t)
        # t = time.time()
        wg = self._interpolate(w_pre, wc, g)
        # print('_interpolate', time.time() - t)
        # t = time.time()
        ww = self._shift(wg, s)
        # print('_shift', time.time() - t)
        # t = time.time()
        w = self._sharpen(ww, gamma)
        # print('_sharpen', time.time() - t)
        # exit()
        return w


class NTMReadHead(nn.Module):

    def __init__(self, memory, cdim):
        super(NTMReadHead, self).__init__()
        N, M = memory.size()
        self.memory = memory
        self.N = N
        self.M = M
        self.read_lens = [M, 1, 1, 3, 1]
        self.fc_read = nn.Linear(cdim, sum(self.read_lens))
        self.init_state = nn.Parameter(torch.zeros(N),
                                       requires_grad=False)
        # self.init_state[0] = 1

    def create_new_state(self, bsz):
        return self.init_state.expand(bsz, self.N)

    def is_read_head(self):
        return True

    def forward(self, hid, w_pre):
        o = self.fc_read(hid)
        k, beta, g, s, gamma = \
            utils.split_cols(o, self.read_lens)
        w = self.memory.address(k, beta, g, s, gamma, w_pre)
        r = self.memory.read(w)

        return r, w


class NTMWriteHead(nn.Module):

    def __init__(self, memory, cdim):
        super(NTMWriteHead, self).__init__()
        N, M = memory.size()
        self.memory = memory
        self.N = N
        self.M = M
        self.write_lens = [M, 1, 1, 3, 1, M, M]
        self.fc_write = nn.Linear(cdim, sum(self.write_lens))
        self.init_state = nn.Parameter(torch.zeros(N),
                                       requires_grad=False)
        # self.init_state[0] = 1

    def create_new_state(self, bsz):
        return self.init_state.expand(bsz, self.N)

    def is_read_head(self):
        return False

    def forward(self, hid, w_pre):
        o = self.fc_write(hid)
        k, beta, g, s, gamma, e, a = \
            utils.split_cols(o, self.write_lens)

        e = torch.sigmoid(e)
        w = self.memory.address(k, beta, g, s, gamma, w_pre)

        self.memory.write(w, e, a)
        return w


class EncoderNTM(MANNBaseEncoder):

    def __init__(self, args):
        idim = args.idim
        cdim = args.hdim
        N = args.N
        M = args.M
        drop = args.dropout
        read_first = args.read_first
        super(EncoderNTM, self).__init__(idim, cdim, N, M, drop, read_first=read_first)
        self.mem = NTMMemory(N, M)
        self.rhead = NTMReadHead(self.mem, cdim)
        self.whead = NTMWriteHead(self.mem, cdim)

    def reset_read(self, bsz):
        self.rstate = self.rhead.create_new_state(bsz)

    def reset_write(self, bsz):
        self.wstate = self.whead.create_new_state(bsz)

    def reset_mem(self, bsz):
        self.mem.reset(bsz)

    def read(self, controller_outp):
        r, self.rstate = self.rhead(controller_outp, self.rstate)

        if 'analysis_mode' in dir(self) and self.analysis_mode:
            assert 'fntm' in dir(self)
            assert self.rstate.shape[0] == 1
            line = {'type': 'read',
                    'w': utils.round_lst(self.rstate[0].cpu().numpy().tolist())}
            line = json.dumps(line)
            # print(line)
            print(line, file=self.fntm)

        return r

    def write(self, controller_outp, input):
        self.wstate = self.whead(controller_outp, self.wstate)

        if 'analysis_mode' in dir(self) and self.analysis_mode:
            assert 'fntm' in dir(self)
            assert self.rstate.shape[0] == 1
            line = {'type': 'write',
                    'w': utils.round_lst(self.wstate[0].cpu().numpy().tolist()),
                    'mem': utils.round_lst2d(self.mem.memory[0].cpu().numpy().tolist())}
            line = json.dumps(line)
            # print(line)
            print(line, file=self.fntm)
