import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from .MANN import MANNBaseEncoder
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack
import json

class EncoderALSTM(MANNBaseEncoder):
    def __init__(self,
                 idim,
                 cdim,
                 N,
                 M,
                 drop,
                 read_first):
        super(EncoderALSTM, self).__init__(idim, cdim, N, M, drop, read_first=read_first)
        self.atten = utils.Attention(cdim, M)
        self.zero = nn.Parameter(torch.zeros(M), requires_grad=False)

    def read(self, controller_outp):
        bsz = controller_outp.shape[0]
        a = None
        if len(self.mem) > 1:
        # mem: (seq_len, bsz, cdim)
            # previous N-1 cells
            # mem = torch.cat(self.mem[:self.N - 1], dim=1)
            mem = torch.cat(self.mem[:len(self.mem) - 1], dim=1)
            c, a = self.atten(controller_outp, mem)
        else:
            c = self.zero.expand(bsz, self.M)

        if 'analysis_mode' in dir(self) and self.analysis_mode:
            assert 'falstm' in dir(self)
            # assert a.shape[0] == 1
            if a is not None:
                line = {'type': 'attention',
                        'a': a[0].cpu().numpy().tolist()}
            else:
                line = {'type': 'attention',
                        'a': []}
            line = json.dumps(line)
            print(line)
            print(line, file=self.falstm)

        return c

    def write(self, controller_outp, r):
        self.mem.append(controller_outp.unsqueeze(1))
        if len(self.mem) > self.N + 1:
            self.mem.pop(0)

    def reset_read(self, bsz):
        pass

    def reset_write(self, bsz):
        pass

    def reset_mem(self, bsz):
        self.mem = []
