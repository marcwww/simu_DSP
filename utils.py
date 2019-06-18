import numpy as np
from torch.nn.init import xavier_uniform_
from torch.nn.init import kaiming_normal_
from torch.nn.init import uniform_
from torch.nn.init import orthogonal_
import torch
from torch import nn
from torch.nn import functional as F
import logging
import random
import time
from torch import optim
from torch.autograd import Variable
from macros import *
import argparse
import sys
from hparams import opts


class DataIter(object):

    def __init__(self, opt, nbatch, gen_batch):
        self.nbatch = nbatch
        self.bidx = 0
        location = opt.gpu if torch.cuda.is_available() and \
                              opt.gpu != -1 else 'cpu'
        self.device = torch.device(location)
        self.gen_batch = gen_batch
        self.opt = opt

    def __iter__(self):
        return self

    def __len__(self):
        return self.nbatch

    def _restart(self):
        self.bidx = 0

    def _gen(self):
        return self.gen_batch(self.opt)

    def __next__(self):
        if self.bidx >= self.nbatch:
            self._restart()
            raise StopIteration()

        inp, outp = self._gen()
        self.bidx += 1
        return inp.to(self.device), outp.to(self.device)


def param_str(opt):
    res_str = {}
    for attr in dir(opt):
        if attr[0] != '_':
            res_str[attr] = getattr(opt, attr)
    to_print = '\n'.join([str(key) + ': ' + str(val) for key, val in res_str.items()])
    logging.info('\n' + to_print)
    return res_str


def mdl_save(model, basename, epoch, loss, valid_perf):
    model_fname = f'{basename}-{epoch}-{loss:.4f}-{str(valid_perf)}.model'
    save_path = os.path.join(MDLS, model_fname)
    logging.info(f'Saving to {save_path}')
    torch.save(model.state_dict(), save_path)


def time_int():
    return int(time.time())


def progress_bar(percent, loss, epoch):
    """Prints the progress until the next report."""

    fill = int(percent * 40)
    str_disp = "\r[%s%s]: %.2f/epoch %d" % ('=' * fill,
                                            ' ' * (40 - fill),
                                            percent,
                                            epoch)
    for k, v in loss.items():
        str_disp += ' (%s:%.4f)' % (k, v)

    print(str_disp, end='')


def parse_opts(description):
    parser = argparse. \
        ArgumentParser(description=description,
                       formatter_class=argparse.
                       ArgumentDefaultsHelpFormatter)
    opts.general_opts(parser)
    if '-task' in sys.argv:
        task = sys.argv[sys.argv.index('-task') + 1]
    else:
        task = parser._option_string_actions['-task'].default

    if '-enc_type' in sys.argv:
        enc_type = sys.argv[sys.argv.index('-enc_type') + 1]
    else:
        enc_type = parser._option_string_actions['-enc_type'].default

    parser = opts.select_opt(task, enc_type, parser)
    opt = parser.parse_args()

    return opt


def select_optim(opt, model):
    if opt.optim == 'rmsprop':
        optimizer = optim.RMSprop(params=filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=opt.lr)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                               lr=opt.lr)
    else:
        raise ModuleNotFoundError

    return optimizer


def init_seed(seed=None):
    def get_ms():
        """Returns the current time in miliseconds."""
        return time.time() * 1000

    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


def init_model(model, method='xavier'):
    if method == 'xavier':
        method = xavier_uniform_
    elif method == 'uniform':
        method = uniform_
    elif method == 'orthogonal':
        method = orthogonal_

    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            method(p)


def build_device(args):
    location = args.gpu if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
    device = torch.device(location)
    return device


def model_loading(opt, model, sub=False):
    model_fname = opt.fload
    location = {'cuda:' + str(opt.gpu): 'cuda:' + str(opt.gpu)} if opt.gpu != -1 else 'cpu'
    model_path = os.path.join(MDLS, model_fname)
    if sub:
        model_path = os.path.join('..', model_path)
    model_dict = torch.load(model_path, map_location=location)
    model.load_state_dict(model_dict)
    print('Loaded from ' + model_path)


def split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


def modulo_convolve(w, s):
    # w: (bsz, N)
    # s: (bsz, 3)
    bsz, ksz = s.shape
    assert ksz == 3

    # t: (1, bsz, 1+N+1)
    t = torch.cat([w[:, -1:], w, w[:, :1]], dim=-1). \
        unsqueeze(0)
    device = s.device
    kernel = torch.zeros(bsz, bsz, ksz).to(device)
    kernel[range(bsz), range(bsz), :] += s
    # c: (bsz, N)
    c = F.conv1d(t, kernel).squeeze(0)
    return c


def bin_vec(num, dim):
    assert type(num) == int
    bin_str = bin(num)[2:]
    assert len(bin_str) < dim
    bin_lst = [0] * (dim - len(bin_str)) + list(map(int, bin_str))
    bin_arr = np.array(bin_lst)
    return bin_arr


def bivec_tensor2int(bivec):
    res = int(''.join(list(map(str, list(bivec.int().numpy())))), 2)
    return res


def round_lst(lst, n=4):
    return list(map(lambda x: round(x, n), lst))


def round_lst2d(lst, n=4):
    return [round_lst(row, n) for row in lst]


class analy(object):

    def __init__(self, model, fnames_dict):
        self.model = model
        self.fnames_dict = fnames_dict

    def __enter__(self):
        self.model.analysis_mode = True
        for name in self.fnames_dict:
            setattr(self.model, name, open(self.fnames_dict[name], 'w'))

    def __exit__(self, *args):
        self.model.analysis_mode = False
        for name in self.fnames_dict:
            f = getattr(self.model, name)
            f.close()


class Attention(nn.Module):
    def __init__(self, cdim, odim):
        super(Attention, self).__init__()
        self.c2r = nn.Linear(cdim, odim)

    def forward(self, h, mem):
        # h: (bsz, hdim)
        # h_current: (bsz, 1, 1, hdim)
        h_current = h.unsqueeze(1).unsqueeze(1)
        # mem: (bsz, len_total, hdim, 1)
        mem = mem.unsqueeze(-1)
        # a: (bsz, len_total, 1, 1)
        a = h_current.matmul(mem)
        a = F.softmax(a, dim=1)
        # c: (bsz, len_total, hdim, 1)
        c = a * mem
        # c: (bsz, hdim)
        c = c.sum(1).squeeze(-1)
        r = self.c2r(c)
        return r, a[:, :, 0, 0]
