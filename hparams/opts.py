from macros import *
from hparams import *


def general_opts(parser):
    group = parser.add_argument_group('general')
    # group.add_argument('-enc_type', type=str, default='lstm')
    # group.add_argument('-enc_type', type=str, default='organics')
    group.add_argument('-enc_type', type=str, default='ntm')
    # group.add_argument('-enc_type', type=str, default='ntmnos')
    # group.add_argument('-enc_type', type=str, default='ntmr')

    group.add_argument('-task', type=str, default='repeat')
    # group.add_argument('-task', type=str, default='mirror')
    group.add_argument('-sub_task', type=str, default='overall')
    # group.add_argument('-optim', type=str, default='rmsprop')
    group.add_argument('-optim', type=str, default='adam')
    group.add_argument('-gpu', type=int, default=-1)
    group.add_argument('--continue_training',
                       action='store_true', default=False)


def select_opt(task, enc_type, parser):
    if task == 'repeat' and enc_type == 'lstm':
        param_file = repeat_lstm
    elif task == 'repeat' and enc_type == 'ntm':
        param_file = repeat_ntm
    elif task == 'repeat' and enc_type == 'sarnn':
        param_file = repeat_sarnn
    elif task == 'repeat' and enc_type == 'sarnnhc':
        param_file = repeat_sarnnhc
    elif task == 'repeat' and enc_type == 'alstm':
        param_file = repeat_alstm
    elif task == 'repeat' and enc_type == 'rnn':
        param_file = repeat_rnn
    elif task == 'mirror' and enc_type == 'lstm':
        param_file = mirror_lstm
    elif task == 'mirror' and enc_type == 'ntm':
        param_file = repeat_ntm
    elif task == 'mirror' and enc_type == 'sarnn':
        param_file = mirror_sarnn
    elif task == 'mirror' and enc_type == 'sarnnhc':
        param_file = mirror_sarnnhc
    elif task == 'mirror' and enc_type == 'alstm':
        param_file = mirror_alstm
    elif task == 'mirror' and enc_type == 'rnn':
        param_file = mirror_rnn
    elif task == 'm10ae' and enc_type == 'sarnnhc':
        param_file = m10ae_sarnnhc
    elif task == 'm10ae' and enc_type == 'ntm':
        param_file = m10ae_ntm
    elif task == 'sort' and enc_type == 'ntm':
        param_file = sort_ntm
    elif task == 'sort' and enc_type == 'sarnnhc':
        param_file = sort_sarnnhc
    elif task == 'repeat_mky' and enc_type == 'lstm':
        param_file = repeat_mky_lstm
    elif task == 'mirror_mky' and enc_type == 'lstm':
        param_file = mirror_mky_lstm
    else:
        raise ModuleNotFoundError
    param_file.model_opts(parser)
    param_file.train_opts(parser)
    return parser
