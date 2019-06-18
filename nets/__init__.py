from nets.LSTM import EncoderLSTM
from nets.NTM import EncoderNTM
from nets.SARNN import EncoderSARNN
from nets.SARNNhc import EncoderSARNNhc
from nets.ALSTM import EncoderALSTM
from nets.SimpRNN import EncoderSRNN


def select_enc(opt):
    enc_type = opt.enc_type
    if enc_type == 'lstm':
        encoder = EncoderLSTM(opt)
    elif enc_type == 'rnn':
        encoder = EncoderSRNN(opt)
    elif enc_type == 'ntm':
        encoder = EncoderNTM(opt)
    elif enc_type == 'sarnn':
        encoder = EncoderSARNN(opt)
    elif enc_type == 'sarnnhc':
        encoder = EncoderSARNNhc(opt)
    elif enc_type == 'alstm':
        encoder = EncoderALSTM(opt)
    else:
        raise ModuleNotFoundError

    return encoder
