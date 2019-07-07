import os
import shutil
import numpy as np
from tqdm import tqdm

from src.navierstokes.gif_data import \
    make_quiver, make_streamplot, make_gif


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_npz', type=str, 
                        help='path to .npz file from model')
    args = parser.parse_args()

    dset = np.load(args.model_npz)
    X, Y, u_seq, v_seq, p_seq = (dset['X'], dset['Y'], dset['u'], dset['v'], dset['p'])
    T = u_seq.shape[0]

    CUR_DIR = os.path.realpath(os.path.dirname(__file__))
    gif_dir = os.path.join(CUR_DIR, 'gifs')
    tmp_dir = os.path.join(gif_dir, 'tmp')

    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    quiver_paths, stream_paths = [], []

    for t in tqdm(range(T)):
        u, v, p = u_seq[t], v_seq[t], p_seq[t]

        quiver_t_path = os.path.join(tmp_dir, 'quiver_cm_t_{}.png'.format(t))
        stream_t_path = os.path.join(tmp_dir, 'stream_cm_t_{}.png'.format(t))

        make_quiver(X, Y, u[0,0], v[0,0], p[0,0], quiver_t_path)
        make_streamplot(X, Y, u[0,0], v[0,0], p[0,0], stream_t_path)

        quiver_paths.append(quiver_t_path)
        stream_paths.append(stream_t_path)

    # convert those into three gifs
    make_gif(quiver_paths, os.path.join(gif_dir, 'quiver_cm.gif'))
    make_gif(stream_paths, os.path.join(gif_dir, 'stream_cm.gif'))

    # delete files
    shutil.rmtree(tmp_dir)
