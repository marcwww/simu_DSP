from macros import *

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-idim', type=int, default=9)
    group.add_argument('-hdim', type=int, default=36)
    group.add_argument('-odim', type=int, default=9 - 1)
    group.add_argument('-N', type=int, default=4)
    # group.add_argument('-N', type=int, default=3)
    group.add_argument('-M', type=int, default=10)
    group.add_argument('-dropout', type=float, default=0)
    group.add_argument('--read_first',
                       action='store_true', default=True)

def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-seed', type=int, default=1000)
    group.add_argument('-nepoch', type=int, default=100)
    group.add_argument('-nbatch_train', type=int, default=10000)
    group.add_argument('-nbatch_valid', type=int, default=100)
    group.add_argument('-valid_times', type=int, default=10)
    group.add_argument('-fload', type=str, default='repeat-overall-ntmr-1547018201.model')
    group.add_argument('-bsz', type=int, default=2)
    group.add_argument('-lr', type=float, default=1e-3)
    group.add_argument('-min_len_train', type=int, default=1)
    group.add_argument('-max_len_train', type=int, default=4)
    group.add_argument('-min_len_valid', type=int, default=4)
    group.add_argument('-max_len_valid', type=int, default=4)
    group.add_argument('-patience', type=int, default=10000)
    group.add_argument('-gclip', type=int, default=15)
