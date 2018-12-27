import numpy as np
from torch.nn.init import xavier_uniform_
from torch.nn.init import kaiming_normal_
import torch
from torch import nn
from torch.nn import functional as F
import logging
import random
import time
from torch.autograd import Variable
from macros import *
import argparse
import sys
from hparams import opts

class DataIter(object):

    def __init__(self, opt, nbatch, gen_batch):
        self.nbatch = nbatch
        self.bidx = 0
        location = opt.device if torch.cuda.is_available() and \
                                 opt.device != -1 else 'cpu'
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
    return res_str

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

def init_model(model, method):
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            method(p)

def model_loading(opt, model):
    model_fname = opt.fload
    location = {'cuda:' + str(opt.gpu): 'cuda:' + str(opt.gpu)} if opt.gpu != -1 else 'cpu'
    model_path = os.path.join(MDLS, model_fname)
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
    t = torch.cat([w[:,-1:], w, w[:,:1]], dim=-1).\
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