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


class EncoderSARNN(MANNBaseEncoder):
    def __init__(self, args):
        idim = args.idim
        cdim = args.hdim
        N = args.N
        M = args.M
        drop = args.dropout
        read_first = args.read_first
        K = args.K
        assert K <= N
        super(EncoderSARNN, self).__init__(idim, cdim, N, M, drop,
                                           read_first=read_first)

        self.K = K
        self.mem_bias = nn.Parameter(torch.zeros(M),
                                     requires_grad=False)
        self.update_kernel = nn.Parameter(torch.eye(N + 1).
                                          view(N + 1, 1, N + 1, 1)[:K+1],
                                          requires_grad=False)

        self.policy_stack = nn.Sequential(nn.Conv1d(M, 2, kernel_size=2),
                                          nn.Linear(N-1, K+1))
        # 2 for push and stay
        # self.policy_input = nn.Linear(idim, 2 * (K + 1))
        self.policy_input = nn.Linear(cdim, 2 * (K + 1))
        self.hid2pushed = nn.Linear(cdim, M)

    def policy(self, input):
        bsz = input.shape[0]
        mem_padded = self.mem.transpose(1, 2)
        # mem_padded = F.pad(self.mem.transpose(1, 2),
        #                    [0, 2], 'constant', 0)
        policy_stack = self.policy_stack(mem_padded).view(bsz, -1)
        policy_input = self.policy_input(input)

        return F.softmax(policy_stack + policy_input, dim=1)

    def update_stack(self,
                     p_push, p_stay, hid):
        bsz = hid.shape[0]

        p_stay = p_stay.unsqueeze(-1).unsqueeze(-1)
        p_push = p_push.unsqueeze(-1).unsqueeze(-1)

        mem_padded = F.pad(self.mem.unsqueeze(1),
                           [0, 0, 0, self.N],
                           'constant', 0)

        # m_stay: (bsz, N+1, N, M)
        m_stay = F.conv2d(mem_padded, self.update_kernel)

        # pushed: (bsz, M)
        pushed = self.hid2pushed(hid)
        # pushed: (bsz, N+1, 1, M)
        pushed_expanded = pushed.unsqueeze(1).unsqueeze(1).\
            expand(bsz, self.K + 1, 1, self.M)
        m_push = torch.cat([pushed_expanded, m_stay[:, :, :-1]], dim=2)
        mem_new_stay = (m_stay * p_stay).sum(dim=1)
        mem_new_push = (m_push * p_push).sum(dim=1)

        # mem_new = (m_stay * p_stay).sum(dim=1) + (m_push * p_push).sum(dim=1)
        mem_new = mem_new_stay + mem_new_push
        self.mem = mem_new
        return mem_new_stay, mem_new_push, m_stay, m_push, pushed

    def read(self, controller_outp):
        r = self.mem[:, 0]
        return r

    def write(self, controller_outp, input):
        # r: (bsz, M)
        # controller_outp: (bsz, cdim)
        # ctrl_info: (bsz, 3 + nstack * M)

        hid = controller_outp
        # policy = self.policy(input)
        policy = self.policy(controller_outp)
        p_push, p_stay = torch.chunk(policy, 2, dim=1)
        mem_stay, mem_push, m_stay, m_push, pushed = \
            self.update_stack(p_push, p_stay, hid)

        if 'analysis_mode' in dir(self) and self.analysis_mode:
            assert 'fsarnn' in dir(self)
            assert policy.shape[0] == 1

            val, pos = torch.topk(policy[0], k=1)
            pos = pos.item()
            val = val.item()
            line = {'type': 'actions',
                    'all': policy[0].cpu().numpy().tolist(),
                    'max_pos': pos,
                    'max_val': val,
                    'mem': self.mem[0].cpu().numpy().tolist()}

            # line['mem_stay'] = mem_stay[0].cpu().numpy().tolist()
            # line['mem_push'] = mem_push[0].cpu().numpy().tolist()
            # line['hid'] = hid[0].cpu().numpy().tolist()
            # line['pushed'] = pushed[0].cpu().numpy().tolist()
            # for i, m_push_i in enumerate(m_push[0]):
            #     line['mem_push_%d' % i] = m_push_i.cpu().numpy().tolist()
            # for i, m_stay_i in enumerate(m_stay[0]):
            #     line['mem_stay_%d' % i] = m_stay_i.cpu().numpy().tolist()

            line = json.dumps(line)
            if pos <= 5:
                # print(line)
                print(line, file=self.fsarnn)
                # print('stay after pop %d times with confidence %.3f' % (pos, val))
                # print('stay after pop %d times with confidence %.3f' % (pos, val), file=self.fanalysis)
            else:
                # print(line)
                print(line, file=self.fsarnn)
                # print('push after pop %d times with confidence %.3f' % (pos - 6, val))
                # print('push after pop %d times with confidence %.3f' % (pos-6, val), file=self.fanalysis)

    def reset_read(self, bsz):
        pass

    def reset_write(self, bsz):
        pass

    def reset_mem(self, bsz):
        self.mem = self.mem_bias.expand(bsz, self.N, self.M)
