import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack

class EncoderSRNN(nn.Module):

    def __init__(self, args):
        idim = args.idim
        cdim = args.hdim
        dropout = 0

        super(EncoderSRNN, self).__init__()
        self.idim = idim
        self.odim = cdim
        self.cdim = cdim
        self.controller = nn.RNN(idim, cdim)
        self.dropout = nn.Dropout(dropout)
        self._reset_controller()

        self.h0 = nn.Parameter(torch.randn(cdim) * 0.05, requires_grad=True)

    def _reset_controller(self):
        for p in self.controller.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.idim + self.cdim))
                nn.init.uniform(p, -stdev, stdev)

    def forward(self, **input):
        embs = input['embs']
        embs = self.dropout(embs)
        bsz = embs.shape[1]

        h = self.h0.expand(1, bsz, self.cdim).contiguous()
        os, h = self.controller(embs, h)

        return os




