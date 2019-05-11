import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
import json


class EncoderLSTM(nn.Module):

    def __init__(self, args):
        idim = args.idim
        cdim = args.cdim
        drop = args.drop
        super(EncoderLSTM, self).__init__()
        self.idim = idim
        self.odim = cdim
        self.cdim = cdim
        self.controller = nn.LSTM(idim, cdim)
        self.dropout = nn.Dropout(drop)
        self._reset_controller()

        self.h0 = nn.Parameter(torch.randn(cdim) * 0.05, requires_grad=True)
        self.c0 = nn.Parameter(torch.randn(cdim) * 0.05, requires_grad=True)

    def _reset_controller(self):
        for p in self.controller.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.idim + self.cdim))
                nn.init.uniform_(p, -stdev, stdev)

    def forward(self, **kwargs):

        def _calc_gates(self, inp, h, c):

            w_ih = self.controller.weight_ih_l0
            w_hh = self.controller.weight_hh_l0
            b_ih = self.controller.bias_ih_l0
            b_hh = self.controller.bias_hh_l0

            w = torch.cat([w_ih, w_hh], dim=-1)
            x = torch.cat([inp, h], dim=-1)
            b = b_ih + b_hh

            out_linear = w.matmul(x.squeeze(0).squeeze(0)) + b
            i, f, g, o = torch.split(out_linear, self.cdim, dim=-1)
            i = F.sigmoid(i)
            f = F.sigmoid(f)
            g = F.tanh(g)
            o = F.sigmoid(o)
            c_new = f * c + i * g
            h_new = o * F.tanh(c_new)

            return i, f, g, o, c_new, h_new

        embs = kwargs['embs']
        embs = self.dropout(embs)
        bsz = embs.shape[1]

        h = self.h0.expand(1, bsz, self.cdim).contiguous()
        c = self.c0.expand(1, bsz, self.cdim).contiguous()

        hs = []
        os = []

        for t, emb in enumerate(embs):
            controller_outp, (h, c) = self.controller(emb.unsqueeze(0), (h, c))

            if 'analysis_mode' in dir(self) and self.analysis_mode:
                assert 'flstm' in dir(self)
                i, f, g, o, c_new, h_new = _calc_gates(self, emb.unsqueeze(0), h, c)
                line = {'type': 'gates',
                        't': t,
                        'i': i.cpu().numpy().tolist(),
                        'f': f.cpu().numpy().tolist(),
                        'g': g.cpu().numpy().tolist(),
                        'o': o.cpu().numpy().tolist()}
                line = json.dumps(line)
                print(line, file=self.flstm)

            o = self.dropout(controller_outp)
            hs.append(h)
            os.append(o)

        os = torch.cat(os, dim=0)
        return os
