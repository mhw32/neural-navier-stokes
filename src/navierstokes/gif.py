"""Make a GIF out of the dataset."""

import os
import shutil
import imageio
import numpy as np

from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D

from src.navierstokes.flow import DATA_DIR

dset_f = np.load(os.path.join(DATA_DIR, 'data_nx_50_ny_50_dt_0.001.npz'))
X_f, Y_f, u_seq_f, v_seq_f, p_seq_f = (dset_f['X'], dset_f['Y'], dataset['u'], 
                                       dset_f['v'], dset_f['p'])
X_cf, Y_cf, u_seq_cf, v_seq_cf, p_seq_cf = spatial_coarsen(
    X_f, Y_f, u_seq_f, v_seq_f, p_seq_f, agg_x=5, agg_y=5)

dset_c = np.load(os.path.join(DATA_DIR, 'data_nx_10_ny_10_dt_0.001.npz'))
X_c, Y_c, u_seq_c, v_seq_c, p_seq_c = (dset_c['X'], dset_c['Y'], dset_c['u'], 
                                       dset_c['v'], dset_c['p'])

T = u_seq_f.shape[0]

# strategy: make a bunch of pngs in a folder, convert
# them to a GIF, and then delete all the images.

def make_quiver(X, Y, u, v, p, save_path):
    fig = plt.figure(figsize=(11, 7), dpi=100)
    plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    plt.colorbar()
    plt.contour(X, Y, p, cmap=cm.viridis)
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(save_path)

def make_streamplot(X, Y, u, v, p, save_path):
    fig = plt.figure(figsize=(11, 7), dpi=100)
    plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    plt.colorbar()
    plt.contour(X, Y, p, cmap=cm.viridis)
    plt.streamplot(X, Y, u, v)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(save_path)

def make_gif(image_paths, gif_path, duration=.1):
    images = [imageio.imread(path) for path in image_paths]
    imageio.mimsave(gif_path, images, duration=duration)

CUR_DIR = os.path.realpath(os.path.dirname(__file__))
gif_dir = os.path.join(CUR_DIR, 'gifs')
tmp_dir = os.path.join(gif_dir, 'tmp')
os.makedirs(gif_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

quiver_f_paths, quiver_c_paths, quiver_cf_paths = [], [], []
stream_f_paths, stream_c_paths, stream_cf_paths = [], [], []

for t in range(T):
    u_f, v_f, p_f = u_seq_f[t], v_seq_f[t], p_seq_f[t]
    u_c, v_c, p_c = u_seq_c[t], v_seq_c[t], p_seq_c[t]
    u_cf, v_cf, p_cf = u_seq_cf[t], v_seq_cf[t], p_seq_cf[t]

    quiver_f_t_path = os.path.join(tmp_dir, 'quiver_f_t_{}.png'.format(t))
    quiver_c_t_path = os.path.join(tmp_dir, 'quiver_c_t_{}.png'.format(t))
    quiver_cf_t_path = os.path.join(tmp_dir, 'quiver_cf_t_{}.png'.format(t))

    stream_f_t_path = os.path.join(tmp_dir, 'stream_f_t_{}.png'.format(t))
    stream_c_t_path = os.path.join(tmp_dir, 'stream_c_t_{}.png'.format(t))
    stream_cf_t_path = os.path.join(tmp_dir, 'stream_cf_t_{}.png'.format(t))

    make_quiver(X_f, Y_f, u_f, v_f, p_f, quiver_f_t_path)
    make_quiver(X_c, Y_c, u_c, v_c, p_c, quiver_c_t_path)
    make_quiver(X_cf, Y_cf, u_cf, v_cf, p_cf, quiver_cf_t_path)

    make_streamplot(X_f, Y_f, u_f, v_f, p_f, stream_f_t_path)
    make_streamplot(X_c, Y_c, u_c, v_c, p_c, stream_c_t_path)
    make_streamplot(X_cf, Y_cf, u_cf, v_cf, p_cf, stream_cf_t_path)

    quiver_f_paths.append(quiver_f_t_path)
    quiver_c_paths.append(quiver_c_t_path)
    quiver_cf_paths.append(quiver_cf_t_path)

    streamplot_f_paths.append(stream_f_t_path)
    streamplot_c_paths.append(stream_c_t_path)
    streamplot_cf_paths.append(stream_cf_t_path)

# convert those into three gifs
make_gif(quiver_f_paths, os.path.join(gif_dir, 'quiver_f.gif'))
make_gif(quiver_c_paths, os.path.join(gif_dir, 'quiver_c.gif'))
make_gif(quiver_cf_paths, os.path.join(gif_dir, 'quiver_cf.gif'))

make_gif(stream_f_paths, os.path.join(gif_dir, 'stream_f.gif'))
make_gif(stream_c_paths, os.path.join(gif_dir, 'stream_c.gif'))
make_gif(stream_cf_paths, os.path.join(gif_dir, 'stream_cf.gif'))

# delete files
shutil.rmtree(tmp_dir)
