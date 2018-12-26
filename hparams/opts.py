from macros import *
from . import repeat_lstm, repeat_organics, repeat_ntm

def general_opts(parser):
    group = parser.add_argument_group('general')
    # group.add_argument('-enc_type', type=str, default='lstm')
    # group.add_argument('-enc_type', type=str, default='organics')
    group.add_argument('-enc_type', type=str, default='ntm')
    group.add_argument('-task', type=str, default='repeat')
    group.add_argument('-sub_task', type=str, default='overall')
    group.add_argument('-optim', type=str, default='rmsprop')
    group.add_argument('-gpu', type=int, default=-1)
    group.add_argument('--continue_training',
                       action='store_true', default=False)

def select_opt(task, enc_type, parser):

    if task == 'repeat' and enc_type == 'lstm':
        repeat_lstm.model_opts(parser)
        repeat_lstm.train_opts(parser)
    elif task == 'repeat' and enc_type == 'organics':
        repeat_organics.model_opts(parser)
        repeat_organics.train_opts(parser)
    elif task == 'repeat' and enc_type == 'ntm':
        repeat_ntm.model_opts(parser)
        repeat_ntm.train_opts(parser)
    else:
        raise ModuleNotFoundError

    return parser