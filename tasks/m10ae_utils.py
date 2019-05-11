import numpy as np
import random
# import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from macros import *
from torch import nn
import utils
import crash_on_ipy

MAX_DEPTH = 20
PROB_BRANCH = 0.4
PROB_PARENTHESIS = 0.3
VALUES = range(1, 10)
NUMERALS = list(map(str, VALUES)) + ['0']
OPS_E = [0, 1]
OPS_T = [2, 3]
NTYPES = ['e', 't']
OP_MAP = ['+', '-', '*', '/']
OPS = OP_MAP
IDX2STR = NUMERALS + OP_MAP
STR2IDX = {symb: idx for idx, symb in enumerate(IDX2STR)}
embedding = torch.eye(len(IDX2STR))
idim = embedding.shape[-1]


def gen_expression(elen, lpo_ratio=0.2):
    assert elen > 2 and elen % 2 == 1
    assert 0 <= lpo_ratio <= 1
    expr = np.random.randint(1, 9, size=elen)
    expr = list(map(lambda x: str(x), expr))
    hop_pos = set(range(1, elen, 2))
    nlpo = int(len(hop_pos) * lpo_ratio)
    lop_pos = []
    for _ in range(nlpo):
        pos = random.choice(list(hop_pos))
        lop_pos.append(pos)
        hop_pos.remove(pos)

    for pos in hop_pos:
        expr[pos] = random.choice(['*', '/'])

    for pos in lop_pos:
        expr[pos] = random.choice(['+', '-'])

    return expr


def to_value(nlst, modd=False):
    def m10eval(op, a0, a1):
        if op == '/':
            res = int(a0) // int(a1) if not modd else int(a0) % int(a1)
        else:
            res = eval(a0 + op + a1)
        res = res % 10
        return str(res)

    def reducible(mem, ninp):
        if len(mem) < 2:
            return False

        top, sec = mem[0], mem[1]
        if top in OPS and sec in NUMERALS:
            return True
        elif top in NUMERALS and sec[0] in NUMERALS and sec[1] in OPS:
            if sec[1] in ['+', '-'] and ninp not in ['*', '/']:
                return True
            if sec[1] in ['*', '/']:
                return True
            return False
        elif top == ')' and sec[0] == '(' and sec[1] in NUMERALS:
            return True
        elif top in NUMERALS and sec == '(' and ninp == ')':
            return True

        return False

    def reduce(mem):
        top = mem.pop(0)
        sec = mem.pop(0)

        if top in OPS and sec in NUMERALS:
            reduced = (sec, top)
        elif top in NUMERALS and sec[0] in NUMERALS and sec[1] in OPS:
            reduced = m10eval(sec[1], sec[0], top)
        elif top == ')' and sec[0] == '(' and sec[1] in NUMERALS:
            reduced = sec[1]
        elif top in NUMERALS and sec == '(':
            reduced = (sec, top)
        else:
            raise NotImplementedError

        return reduced

    stack = []
    reduce_lst = []
    for t, n in enumerate(nlst):
        stack.insert(0, n)
        # reduce_lst.append(0)
        if t != len(nlst) - 1:
            ninp = nlst[t + 1]
        else:
            ninp = None

        r = 0
        while reducible(stack, ninp):
            stack.insert(0, reduce(stack))
            r += 1
            # reduce_lst.append(1)
        reduce_lst.append(r)

    return stack[0], reduce_lst


def exprs2tensor(exprs):
    exprs = [list(map(lambda x: STR2IDX[x], expr)) for expr in exprs]
    indices = torch.LongTensor(exprs)
    tensor = F.embedding(indices, embedding)
    return tensor


def gen_batch(min_len, max_len, min_lopr, max_lopr, modd, bsz, batch_first=False):
    exprs = []
    vals = []
    elen = random.randint(min_len, max_len)
    if elen % 2 != 1:
        elen += 1
    lopr = random.uniform(min_lopr, max_lopr)
    while len(exprs) != bsz:
        expr = gen_expression(elen, lopr)
        try:
            val, _ = to_value(expr, modd)
            exprs.append(expr)
            vals.append(int(val))
        except:
            pass

    # inp: (bsz, seq_len, idim=14)
    # tar: (bsz, nclasses=10)
    inp = exprs2tensor(exprs).float()
    tar = torch.Tensor(vals).long()
    if not batch_first:
        inp = inp.transpose(0, 1)
    return inp, tar


if __name__ == '__main__':
    expr = gen_expression(11, 0.2)
    print(expr)
    val = to_value(expr)
    print(val)

    print(gen_batch(2, 20, 0.1, 0.5, False, 2))
