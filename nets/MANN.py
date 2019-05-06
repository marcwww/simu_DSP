import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack
import json


class MANNBaseEncoder(nn.Module):

    def __init__(self, idim, cdim, N, M, dropout, read_first):
        super(MANNBaseEncoder, self).__init__()
        self.idim = idim
        self.odim = cdim + M
        self.cdim = cdim
        self.N = N
        self.M = M
        self.controller = nn.LSTM(idim + M, cdim)
        self.dropout = nn.Dropout(dropout)
        self._reset_controller()
        self.read_first = read_first

        self.h0 = nn.Parameter(torch.randn(1, 1, cdim) * 0.05, requires_grad=True)
        self.c0 = nn.Parameter(torch.randn(1, 1, cdim) * 0.05, requires_grad=True)
        self.r0 = nn.Parameter(torch.randn(1, M) * 0.02, requires_grad=False)

    def _reset_controller(self):
        for p in self.controller.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.idim + self.M + self.cdim))
                nn.init.uniform(p, -stdev, stdev)

    def reset_read(self, bsz):
        raise NotImplementedError

    def reset_write(self, bsz):
        raise NotImplementedError

    def reset_mem(self, bsz):
        raise NotImplementedError

    def read(self, controller_outp):
        raise NotImplementedError

    def write(self, controller_outp, input):
        raise NotImplementedError

    def forward(self, **input):

        def _calc_gates(self, inp, h, c):

            w_ih = self.controller.weight_ih_l0
            w_hh = self.controller.weight_hh_l0
            b_ih = self.controller.bias_ih_l0
            b_hh = self.controller.bias_hh_l0

            w = torch.cat([w_ih, w_hh], dim=-1)
            x = torch.cat([inp, h], dim=-1)
            b = b_ih + b_hh

            out_linear = w.matmul(x.squeeze(0).squeeze(0)) + b

            # out_linear = F.linear(x, w, b)
            i, f, g, o = torch.split(out_linear, self.cdim, dim=-1)
            i = F.sigmoid(i)
            f = F.sigmoid(f)
            g = F.tanh(g)
            o = F.sigmoid(o)
            c_new = f * c + i * g
            h_new = o * F.tanh(c_new)

            return i, f, g, o, c_new, h_new

        embs = input['embs']
        embs = self.dropout(embs)
        bsz = embs.shape[1]

        self.reset_read(bsz)
        self.reset_write(bsz)
        self.reset_mem(bsz)

        h = self.h0.repeat(1, bsz, 1)
        c = self.c0.repeat(1, bsz, 1)
        r = self.r0.repeat(bsz, 1)

        hs = []
        cs = []
        os = []
        for t, emb in enumerate(embs):
            controller_inp = torch.cat([emb, r], dim=1).unsqueeze(0)
            if 'analysis_mode' in dir(self) and self.analysis_mode:
                assert 'flstm' in dir(self)
                i, f, g, o, c_new, h_new = _calc_gates(self, controller_inp, h, c)
                line = {'type': 'gates',
                        't': t,
                        'i': i.cpu().numpy().tolist(),
                        'f': f.cpu().numpy().tolist(),
                        'g': g.cpu().numpy().tolist(),
                        'o': o.cpu().numpy().tolist()}
                line = json.dumps(line)
                print(line)
                print(line, file=self.flstm)

            controller_outp, (h, c) = self.controller(controller_inp, (h, c))
            controller_outp = controller_outp.squeeze(0)

            if self.read_first:
                r = self.read(controller_outp)
                self.write(controller_outp, emb)
            else:
                self.write(controller_outp, emb)
                r = self.read(controller_outp)

            o = torch.cat([controller_outp, r], dim=1)
            o = self.dropout(o)

            hs.append(h)
            cs.append(c)
            os.append(o.unsqueeze(0))

        os = torch.cat(os, dim=0)
        cs = torch.cat(cs, dim=0)
        # np.savetxt('sarnn_cells.txt', cs[:, 0].cpu().numpy())

        return os

# TODO: two heads for NTM
# TODO: read top two for SARNN
# TODO: recursive SARNN
