import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
import json

class EncoderORGaNICs(nn.Module):

    def __init__(self, idim, cdim, drop):
        super(EncoderORGaNICs, self).__init__()
        self.idim = idim
        self.odim = cdim
        self.cdim = cdim
        self.dropout = nn.Dropout(drop)

        self.h0 = nn.Parameter(torch.randn(cdim) * 0.05)
        self.h2h = nn.Linear(cdim, cdim)
        self.x2z = nn.Linear(idim, cdim)
        self.w_ax = nn.Parameter(torch.randn(idim, cdim))
        self.w_ah = nn.Parameter(torch.randn(cdim, cdim))
        self.c_a = nn.Parameter(torch.randn(cdim))
        self.w_bx = nn.Parameter(torch.randn(idim, cdim))
        self.w_bh = nn.Parameter(torch.randn(cdim, cdim))
        self.c_b = nn.Parameter(torch.randn(cdim))

    def update(self, x, h):
        z = self.x2z(x)
        h_hat = self.h2h(h)
        a = x.matmul(self.w_ax) + h.matmul(self.w_ah) + self.c_a
        b = x.matmul(self.w_bx) + h.matmul(self.w_bh) + self.c_b

        a_plus = F.relu(a)
        b_plus = F.relu(b)
        h_new = b_plus/(1+b_plus) * z + 1/(1+a_plus) * h_hat
        return h_new

    def forward(self, **kwargs):

        embs = kwargs['embs']
        embs = self.dropout(embs)
        bsz = embs.shape[1]

        h = self.h0.expand(bsz, self.cdim).contiguous()
        os = []

        for t, emb in enumerate(embs):
            # emb: (bsz, idim)
            h = self.update(emb, h)
            o = self.dropout(h)
            os.append(o.unsqueeze(0))

        os = torch.cat(os, dim=0)
        return os
