import os
import torch
import shutil
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_DIR = os.path.realpath(MODEL_DIR)


def spatial_coarsen(X, Y, u_seq, v_seq, p_seq, agg_x=4, agg_y=4):
    """Given dynamics of a certain coarseness, we want to 
    aggregate by averaging over regions in the spatial grid.

    Args
    ----
    X := np.array (size: nx by ny)
         meshgrid for x 
    Y := np.array (size: nx by ny)
         meshgrid for y
    u_seq := np.array (size: T x nx by ny)
             u-momentum components
    v_seq := np.array (size: T x nx by ny)
             v-momentum components
    p_seq := np.array (size: T x nx by ny)
             pressure components
    agg_x := integer (default: 4)
             coarsen factor for x-coordinates
    agg_y := integer (default: 4)
             coarsen factor for y-coordinates

    We return each element but coarsened.
    """
    nx, ny = X.shape[0], X.shape[1]
    T = u_seq.shape[0]

    assert nx % agg_x == 0
    assert ny % agg_y == 0

    new_u_seq = np.zeros((T, nx // agg_x, ny // agg_y))
    new_v_seq = np.zeros((T, nx // agg_x, ny // agg_y))
    new_p_seq = np.zeros((T, nx // agg_x, ny // agg_y))

    new_x = np.linspace(0, 2, nx // agg_x)
    new_y = np.linspace(0, 2, ny // agg_y)
    new_X, new_Y = np.meshgrid(new_x, new_y)

    for i in range(nx // agg_x):
        for j in range(ny // agg_x):
            u_sub = u_seq[:, i*agg_x:(i+1)*agg_x, j*agg_y:(j+1)*agg_y].reshape(T, -1)
            v_sub = v_seq[:, i*agg_x:(i+1)*agg_x, j*agg_y:(j+1)*agg_y].reshape(T, -1)
            p_sub = p_seq[:, i*agg_x:(i+1)*agg_x, j*agg_y:(j+1)*agg_y].reshape(T, -1)

            new_u_seq[:, i, j] = np.mean(u_sub, axis=1)
            new_v_seq[:, i, j] = np.mean(v_sub, axis=1)
            new_p_seq[:, i, j] = np.mean(p_sub, axis=1)

    return new_X, new_Y, new_u_seq, new_v_seq, new_p_seq


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))
