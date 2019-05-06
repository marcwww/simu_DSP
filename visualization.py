import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import utils
np.set_printoptions(precision=3)

def policy(inp_len, N, fname_ntm, title=None):

    seq_len = 2 * inp_len - 1
    fanaly_ntm = fname_ntm
    wr = {i:[] for i in range(seq_len)}
    ww = {i:[] for i in range(seq_len)}

    i = 0
    nc = 0
    nt = 0
    with open(fanaly_ntm, 'r') as f:
        for line in f:
            line_json = json.loads(line)
            if 'w' not in line_json:
                nc += line_json['is_correct']
                nt += 1
                continue

            if line_json['type'] == 'read':
                wr[i].append(line_json['w'])
                ww[i].append([0] * len(line_json['w']))

            if line_json['type'] == 'write':
                ww[i].append(line_json['w'])
                wr[i].append([0] * len(line_json['w']))

            i += 1
            i %= seq_len
    print('correct prediction%:', nc/nt)
    for i in range(seq_len):
        wr[i] = np.array(wr[i]).sum(0)/len(wr[i])
        ww[i] = np.array(ww[i]).sum(0)/len(ww[i])
    wrs = np.array([wr[i] for i in range(seq_len)])
    wws = np.array([ww[i] for i in range(seq_len)])
    ws = np.hstack([wws, np.zeros((seq_len, 1)), wrs])
    fig, ax = plt.subplots()
    fig.dpi = 100
    plt.imshow(ws.transpose(), cmap=plt.cm.Blues)
    plt.yticks(np.arange(0, 2*N+1), [str(i) for i in range(N)] + [''] + [str(i) for i in range(N)], fontsize=20)
    plt.xticks(np.arange(0, seq_len), fontsize=20)
    plt.grid(linewidth=0.2)
    # plt.xticks(np.arange(0, seq_len), inp + tar, fontsize=8)
    plt.ylabel('memory cell position')
    plt.xlabel('time step')
    if title is None:
        plt.title('top write, bottom read', fontsize=10)
    else:
        plt.title('top write, bottom read, pattern %s' % title, fontsize=10)
    fig_name = '../res/figs/' + title + '_' + str(utils.time_int()) + '.png'
    plt.savefig(fig_name)
    plt.show()
